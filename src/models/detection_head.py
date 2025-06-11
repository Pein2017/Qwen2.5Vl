import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """
    Detection head for open vocabulary dense object captioning.

    This module implements DETR-style object detection with caption generation
    capabilities, designed to work with Qwen2.5-VL's hidden states.

    Key Features:
    - Object queries for DETR-style detection
    - Cross-attention decoder to process LLM hidden states
    - Bounding box regression with normalized coordinates
    - Object presence/confidence prediction
    - Caption generation using same vocabulary as base LLM
    """

    def __init__(
        self,
        config,
        num_queries: int,
        max_caption_length: int,
        tokenizer,
        detection_decoder_nhead: int,
        detection_decoder_dim_feedforward_factor: float,
        detection_decoder_num_layers: int,
        detection_caption_decoder_nhead: int,
        detection_caption_decoder_dim_feedforward_factor: float,
        detection_caption_decoder_num_layers: int,
        detection_head_dropout: float,
        dtype=None,
    ):
        super().__init__()
        # Use official config dimensions
        self.hidden_size = config.hidden_size  # 8192 for 72B, 3584 for 7B
        self.num_queries = num_queries
        self.max_caption_length = max_caption_length
        self.vocab_size = config.vocab_size  # 152064
        self.tokenizer = tokenizer

        # Determine target dtype
        if dtype is None:
            dtype = torch.float32  # Default fallback
        self.target_dtype = dtype

        # Object queries for DETR-style detection
        self.object_queries = nn.Embedding(num_queries, self.hidden_size)

        # Token embedding will be set from base model (no separate embedding needed)
        self.token_embedding = None

        # Cross-attention decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=detection_decoder_nhead,
            dim_feedforward=int(
                self.hidden_size * detection_decoder_dim_feedforward_factor
            ),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=detection_decoder_num_layers
        )

        # Bounding box prediction head
        self.bbox_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.SiLU(),
            nn.Dropout(detection_head_dropout),
            nn.Linear(self.hidden_size // 4, 4),
        )

        # Object presence/confidence head
        self.objectness_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.SiLU(),
            nn.Dropout(detection_head_dropout),
            nn.Linear(self.hidden_size // 4, 1),
        )

        # Caption generation head - generates tokens autoregressively
        self.caption_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Dropout(detection_head_dropout),
            nn.Linear(self.hidden_size, self.vocab_size),  # Predict vocabulary tokens
        )

        # Caption decoder for autoregressive generation
        self.caption_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=detection_caption_decoder_nhead,
                dim_feedforward=int(
                    self.hidden_size * detection_caption_decoder_dim_feedforward_factor
                ),
                batch_first=True,
                norm_first=True,
                activation="gelu",
            ),
            num_layers=detection_caption_decoder_num_layers,
        )

        # Use proper special tokens from tokenizer if available
        if tokenizer is not None:
            # Use existing special tokens from the tokenizer
            self.start_token_id = (
                tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
            )
            self.end_token_id = (
                tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
            )
            self.pad_token_id = (
                tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            )
        else:
            raise ValueError("Tokenizer is required for caption generation")

        # Initialize with small weights
        self._init_weights()

        # Convert to target dtype after initialization
        if dtype != torch.float32:
            self.to(dtype)

    def set_token_embedding(self, token_embedding):
        """Set token embedding from base model"""
        self.token_embedding = token_embedding

    def _init_weights(self):
        """Initialize detection head with small weights"""
        # Initialize object queries with small random values
        nn.init.normal_(self.object_queries.weight, std=0.01)

        # Token embedding will be shared from base model, no initialization needed

        # Initialize all heads with small weights
        for module in [self.objectness_head, self.caption_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        # Special initialization for bbox head to prevent saturation
        for i, layer in enumerate(self.bbox_head):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    if i == len(self.bbox_head) - 1:  # Last layer (output layer)
                        # Initialize bias to predict boxes around center with reasonable size
                        # Format: [x1, y1, x2, y2] -> bias for [0.25, 0.25, 0.75, 0.75] after sigmoid
                        # sigmoid^-1(0.25) ≈ -1.1, sigmoid^-1(0.75) ≈ 1.1
                        nn.init.constant_(layer.bias, 0.0)  # Start with center bias
                    else:
                        nn.init.constant_(layer.bias, 0)

    def reinitialize_bbox_head(self):
        """Reinitialize bbox head weights to fix saturation issues"""
        print("🔄 Reinitializing bbox head weights to fix saturation...")

        for i, layer in enumerate(self.bbox_head):
            if isinstance(layer, nn.Linear):
                # Use Xavier/Glorot initialization for better gradient flow
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    if i == len(self.bbox_head) - 1:  # Last layer (output layer)
                        # Initialize bias to predict reasonable boxes
                        # We want raw outputs around 0 so sigmoid gives ~0.5
                        nn.init.constant_(layer.bias, 0.0)
                    else:
                        nn.init.constant_(layer.bias, 0.0)

        print("✅ Bbox head weights reinitialized successfully")

    def forward(
        self, hidden_states, attention_mask, ground_truth_objects=None, training=True
    ):
        """
        Forward pass for detection head.

        Args:
            hidden_states: (B, S, hidden_size) - final LLM hidden states
            attention_mask: (B, S) - to mask padded tokens
            ground_truth_objects: List[List[Dict]] - GT objects for training
            training: bool - whether in training mode

        Returns:
            Dict containing predictions and features
        """
        # hidden_states: (B, S, hidden_size) - final LLM hidden states
        # attention_mask: (B, S) - to mask padded tokens
        # ground_truth_objects: List[List[Dict]] - GT objects for training

        B, S, D = hidden_states.shape

        # Ensure all detection head components match input dtype
        input_dtype = hidden_states.dtype
        if self.object_queries.weight.dtype != input_dtype:
            # Convert all modules to the correct dtype
            self.to(dtype=input_dtype)

        # Object queries
        queries = self.object_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        # Cross-attention: queries attend to LLM hidden states
        memory_key_padding_mask = (
            ~attention_mask if attention_mask is not None else None
        )

        decoded_queries = self.decoder(
            tgt=queries,  # (B, N, D)
            memory=hidden_states,  # (B, S, D)
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # Bbox and objectness predictions
        pred_boxes_raw = self.bbox_head(decoded_queries)  # (B, N, 4)
        pred_boxes = torch.sigmoid(
            pred_boxes_raw
        )  # Apply sigmoid to ensure [0,1] range
        pred_objectness = self.objectness_head(decoded_queries)  # (B, N, 1)

        # Caption generation
        if training and ground_truth_objects is not None:
            # Teacher forcing during training
            caption_logits = self._generate_captions_teacher_forcing(
                decoded_queries, ground_truth_objects
            )
        else:
            # Autoregressive generation during inference
            caption_logits = self._generate_captions_autoregressive(decoded_queries)

        return {
            "pred_boxes": pred_boxes,
            "pred_boxes_raw": pred_boxes_raw,  # Add raw predictions for debugging
            "pred_objectness": pred_objectness.squeeze(-1),  # (B, N)
            "caption_logits": caption_logits,  # (B, N, max_len, vocab_size)
            "object_features": decoded_queries,
        }

    def _generate_captions_teacher_forcing(self, object_features, ground_truth_objects):
        """Generate captions using teacher forcing during training"""
        B, N, D = object_features.shape
        device = object_features.device
        dtype = object_features.dtype

        if self.tokenizer is None:
            return self._generate_captions_fallback(object_features)

        # For training, we'll generate all positions at once but with proper causal masking
        # This is more efficient than true autoregressive generation during training

        # Create a simple causal mask for the caption decoder
        causal_mask = torch.triu(
            torch.ones(
                self.max_caption_length,
                self.max_caption_length,
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )  # Upper triangular mask

        # Create position embeddings for caption positions
        position_ids = torch.arange(self.max_caption_length, device=device)
        position_embeddings = torch.sin(
            position_ids.float().unsqueeze(-1)
            / 10000.0 ** (torch.arange(D, device=device).float() / D)
        ).to(dtype)  # (max_len, D) - match input dtype

        # Expand object features for all caption positions
        # (B, N, D) -> (B, N, max_len, D)
        expanded_features = object_features.unsqueeze(2).expand(
            B, N, self.max_caption_length, D
        )

        # Add position embeddings
        caption_features = expanded_features + position_embeddings.unsqueeze(
            0
        ).unsqueeze(0)

        # Reshape for processing: (B*N, max_len, D)
        caption_features = caption_features.reshape(B * N, self.max_caption_length, D)

        # Apply causal attention using the caption decoder
        # Use self-attention with causal masking
        decoded_features = self.caption_decoder(
            tgt=caption_features,
            memory=caption_features,  # Self-attention
            tgt_mask=causal_mask,
        )  # (B*N, max_len, D)

        # Generate logits for each position
        caption_logits = self.caption_head(
            decoded_features
        )  # (B*N, max_len, vocab_size)

        # Reshape back: (B, N, max_len, vocab_size)
        caption_logits = caption_logits.reshape(B, N, self.max_caption_length, -1)

        return caption_logits

    def _generate_captions_fallback(self, object_features):
        """Fallback caption generation when no tokenizer is available"""
        B, N, D = object_features.shape
        device = object_features.device
        dtype = object_features.dtype

        # Create position embeddings
        position_ids = torch.arange(self.max_caption_length, device=device)
        position_embeddings = torch.sin(
            position_ids.float().unsqueeze(-1)
            / 10000.0 ** (torch.arange(D, device=device).float() / D)
        ).to(dtype)  # (max_len, D) - match input dtype

        # Expand and add position embeddings
        expanded_features = object_features.unsqueeze(2).expand(
            B, N, self.max_caption_length, D
        )
        caption_features = expanded_features + position_embeddings.unsqueeze(
            0
        ).unsqueeze(0)

        # Generate logits
        caption_logits = self.caption_head(
            caption_features
        )  # (B, N, max_len, vocab_size)

        return caption_logits

    def _generate_captions_autoregressive(self, object_features):
        """Generate captions autoregressively during inference"""
        B, N, D = object_features.shape
        device = object_features.device
        dtype = object_features.dtype

        # Initialize with start tokens
        generated_sequences = torch.full(
            (B, N, 1), self.start_token_id, device=device, dtype=torch.long
        )

        # Store logits for each position
        all_logits = []

        for pos in range(self.max_caption_length):
            current_length = generated_sequences.shape[2]

            # Create position embeddings for current sequence
            position_ids = torch.arange(current_length, device=device)
            position_embeddings = torch.sin(
                position_ids.float().unsqueeze(-1)
                / 10000.0 ** (torch.arange(D, device=device).float() / D)
            ).to(dtype)  # (current_length, D) - match input dtype

            # Get token embeddings for generated sequence
            if self.token_embedding is not None:
                # Use shared token embedding from base model
                token_embeds = self.token_embedding(
                    generated_sequences
                )  # (B, N, current_length, D)
            else:
                # Fallback: use object features repeated
                token_embeds = object_features.unsqueeze(2).expand(
                    B, N, current_length, D
                )

            # Add position embeddings
            sequence_features = token_embeds + position_embeddings.unsqueeze(
                0
            ).unsqueeze(0)

            # Reshape for decoder: (B*N, current_length, D)
            sequence_features = sequence_features.reshape(B * N, current_length, D)

            # Create object context for cross-attention
            object_context = object_features.unsqueeze(1).expand(
                B, current_length, N, D
            )
            object_context = (
                object_context.transpose(1, 2)
                .contiguous()
                .reshape(B * N, current_length, D)
            )

            # Apply caption decoder with cross-attention to object features
            decoded_features = self.caption_decoder(
                tgt=sequence_features,
                memory=object_context,
            )  # (B*N, current_length, D)

            # Get logits for the last position (next token prediction)
            last_features = decoded_features[:, -1, :]  # (B*N, D)
            next_logits = self.caption_head(last_features)  # (B*N, vocab_size)

            # Reshape back: (B, N, vocab_size)
            next_logits = next_logits.reshape(B, N, -1)
            all_logits.append(next_logits)

            # Sample next token (greedy decoding)
            next_tokens = next_logits.argmax(dim=-1, keepdim=True)  # (B, N, 1)

            # Append to generated sequences
            generated_sequences = torch.cat([generated_sequences, next_tokens], dim=-1)

            # Check for early stopping (all sequences have EOS)
            if pos > 0 and (next_tokens.squeeze(-1) == self.end_token_id).all():
                break

        # Pad remaining positions if needed
        while len(all_logits) < self.max_caption_length:
            pad_logits = torch.full_like(all_logits[0], -float("inf"))
            pad_logits[..., self.pad_token_id] = (
                0.0  # Only pad token has reasonable probability
            )
            all_logits.append(pad_logits)

        # Stack to get (B, N, max_len, vocab_size)
        caption_logits = torch.stack(all_logits, dim=2)

        return caption_logits

    def generate_single_query_caption(
        self, object_features, query_idx, max_length=None
    ):
        """
        Generate caption for a single query autoregressively.
        This method is designed for the autoregressive generation in simple_demo.py

        Args:
            object_features: (B, N, D) - Object features from decoder
            query_idx: int - Index of the query to generate caption for
            max_length: int - Maximum caption length (default: self.max_caption_length)

        Returns:
            Dict with generation info
        """
        if max_length is None:
            max_length = self.max_caption_length

        B, N, D = object_features.shape
        device = object_features.device
        dtype = object_features.dtype

        # Extract features for the specific query
        query_features = object_features[:, query_idx : query_idx + 1, :]  # (B, 1, D)

        # Initialize with start token
        generated_tokens = [self.start_token_id]
        generation_logits = []

        for step in range(max_length):
            # Create current sequence
            current_seq = torch.tensor([generated_tokens], device=device).unsqueeze(
                0
            )  # (1, 1, len)
            current_length = len(generated_tokens)

            # Get token embeddings
            if self.token_embedding is not None:
                token_embeds = self.token_embedding(current_seq)  # (1, 1, len, D)
            else:
                # Fallback: repeat query features
                token_embeds = query_features.unsqueeze(2).expand(
                    1, 1, current_length, D
                )

            # Add position embeddings
            position_ids = torch.arange(current_length, device=device)
            position_embeddings = torch.sin(
                position_ids.float().unsqueeze(-1)
                / 10000.0 ** (torch.arange(D, device=device).float() / D)
            ).to(dtype)  # Match input dtype

            sequence_features = token_embeds.squeeze(1) + position_embeddings.unsqueeze(
                0
            )  # (1, len, D)

            # Apply decoder
            decoded_features = self.caption_decoder(
                tgt=sequence_features,
                memory=query_features,  # (B, 1, D) -> cross-attention context
            )

            # Get next token logits
            next_logits = self.caption_head(
                decoded_features[:, -1, :]
            )  # (1, vocab_size)
            generation_logits.append(next_logits.squeeze(0))  # (vocab_size,)

            # Sample next token
            next_token = next_logits.argmax(dim=-1).item()
            generated_tokens.append(next_token)

            # Stop conditions
            if next_token == self.end_token_id or next_token == self.pad_token_id:
                break

        return {
            "tokens": generated_tokens,
            "logits": generation_logits,
            "length": len(generated_tokens),
        }

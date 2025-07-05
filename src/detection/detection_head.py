from typing import TypeVar

import torch
import torch.nn as nn
from typeguard import typechecked

from src.config import config as global_config
from .detection_adapter import DetectionAdapter
from src.utils.schema import (  
    AttentionMaskType,
    CaptionLogitsType,
    DetectionPredictions,
    FlattenVisionFeatType,
    LLMTokenType,
    ObjectQueriesType,
    VisionFeatType,
    assert_tensor_shape,
    assert_vision_features,
)

# Dim symbols shared across the codebase
S = TypeVar("S")  # Sequence length
B = TypeVar("B")  # Batch size
N = TypeVar("N")  # Num queries
D = TypeVar("D")  # Hidden dimension


class DetectionHead(nn.Module):
    """
    Detection head for open vocabulary dense object captioning.

    This module implements DETR-style object detection with caption generation
    capabilities, designed to work with Qwen2.5-VL's hidden states.

    Key Features:
    - Object queries for DETR-style detection
    - Cross-attention decoder to process LLM hidden states
    - Bounding bbox_2d regression with normalized coordinates
    - Object presence/confidence prediction
    - Caption generation using same vocabulary as base LLM
    """

    def __init__(
        self,
        config,
        num_queries: int,
        max_caption_length: int,
        tokenizer,
        detection_decoder_dim_feedforward_factor: float,
        detection_decoder_num_layers: int,
        detection_caption_decoder_dim_feedforward_factor: float,
        detection_caption_decoder_num_layers: int,
        detection_head_dropout: float,
        adapter_bottleneck_ratio: int,
        adapter_num_layers: int,
        dtype=None,
    ):
        super().__init__()
        # Derive attention head counts from the base model config
        detection_decoder_nhead = config.num_attention_heads
        detection_caption_decoder_nhead = config.num_attention_heads
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

        # Adapter module to reshape hidden states before detection decoding
        bottleneck = self.hidden_size // adapter_bottleneck_ratio
        self.adapter = DetectionAdapter(
            hidden_size=self.hidden_size,
            bottleneck=bottleneck,
            num_layers=adapter_num_layers,
        )
        # Adapter module for raw vision features from the vision tower
        self.vision_adapter = DetectionAdapter(
            hidden_size=self.hidden_size,
            bottleneck=bottleneck,
            num_layers=adapter_num_layers,
        )

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

        # Bounding bbox_2d prediction head
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

        # Initialize bbox head with Xavier for diverse predictions and zero bias
        for layer in self.bbox_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

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

    @assert_tensor_shape
    @typechecked
    def forward(
        self,
        hidden_states: LLMTokenType | list[LLMTokenType],
        attention_mask: AttentionMaskType,
        vision_feats: VisionFeatType | FlattenVisionFeatType,
        ground_truth_objects: list | None = None,
        training: bool = True,
    ) -> DetectionPredictions:
        """
        Forward pass for detection head.

        Args:
            hidden_states: (B, S, hidden_size) - final LLM hidden states
            attention_mask: (B, S) - to mask padded tokens
            vision_feats: (B, S, hidden_size) - optional vision features
            ground_truth_objects: List[List[Dict]] - GT objects for training
            training: bool - whether in training mode

        Returns:
            DetectionPredictions dataclass containing predictions and features
        """
        # hidden_states: (B, S, hidden_size) - final LLM hidden states
        # attention_mask: (B, S) - to mask padded tokens
        # vision_feats: (B, S, hidden_size) - optional vision features
        # ground_truth_objects: List[List[Dict]] - GT objects for training
        # training: bool - whether in training mode

        # If a list of per-layer hidden states is passed, pick the configured layer for bbox_2d supervision
        if isinstance(hidden_states, (list, tuple)):
            layer_idx = getattr(global_config, "detection_feature_layer", -1)
            box_memory = hidden_states[layer_idx]
        else:
            box_memory = hidden_states

        B, S, D = box_memory.shape  # type: ignore

        # Ensure all detection head components match input dtype
        input_dtype = box_memory.dtype
        if self.object_queries.weight.dtype != input_dtype:
            # Convert all modules to the correct dtype
            self.to(dtype=input_dtype)

        # Object queries
        queries = self.object_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        # Cross-attention: queries attend to LLM hidden states
        # Create key padding mask: True for positions to mask (padding tokens)
        if attention_mask is not None:
            # attention_mask: 1 for real tokens, 0 for padding
            memory_key_padding_mask = ~attention_mask.to(torch.bool)
        else:
            memory_key_padding_mask = None

        # Prepare two-stream memory: language-adapted + optional vision-adapted
        lang_adapted = self.adapter(box_memory)
        # vision_feats must be provided (runtime enforced)
        assert_vision_features(vision_feats)
        # Ensure vision_feats has matching dtype
        if vision_feats.dtype != lang_adapted.dtype:
            vision_feats = vision_feats.to(dtype=lang_adapted.dtype)

        # Adapt raw vision features
        vis_adapted = self.vision_adapter(vision_feats)
        # If single batch instance, add batch dim
        if vis_adapted.ndim == 2:
            vis_adapted = vis_adapted.unsqueeze(0)  # (1, SV, D)
        # If batch dimension mismatches, expand for each batch sample
        if vis_adapted.size(0) != B:
            vis_adapted = vis_adapted.expand(B, -1, -1)

        # vision tokens are never masked
        vis_len = vis_adapted.size(1)
        vis_mask = torch.zeros(
            (B, vis_len), dtype=torch.bool, device=vis_adapted.device
        )

        # Concatenate vision then language memory
        memory = torch.cat([vis_adapted, lang_adapted], dim=1)
        memory_key_padding_mask = torch.cat([vis_mask, memory_key_padding_mask], dim=1)
        decoded_queries: ObjectQueriesType = self.decoder(
            tgt=queries,
            memory=memory,
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
            caption_logits: CaptionLogitsType = self._generate_captions_teacher_forcing(
                decoded_queries, ground_truth_objects
            )
        else:
            # Autoregressive generation during inference
            caption_logits: CaptionLogitsType = self._generate_captions_autoregressive(
                decoded_queries
            )

        # Wrap into structured dataclass (performs its own validation)
        preds = DetectionPredictions(
            pred_boxes=pred_boxes,
            pred_boxes_raw=pred_boxes_raw,
            pred_objectness=pred_objectness.squeeze(-1),
            caption_logits=caption_logits,
            object_features=decoded_queries,
        )

        return preds

    def _generate_captions_teacher_forcing(self, object_features, ground_truth_objects):
        """Generate caption logits with *true* teacher forcing so that the
        caption decoder sees exactly the same type of token embeddings it will
        encounter at inference time.  For each object-query we feed the
        (shifted-right) ground-truth caption tokens if the query is expected
        to correspond to a ground-truth object; otherwise we feed a single EOS
        token followed by PAD.  This eliminates the train/inference mismatch
        where the previous implementation re-used object feature vectors as
        token embeddings."""

        B, N, D = object_features.shape
        device = object_features.device
        dtype = object_features.dtype

        assert self.tokenizer is not None, (
            "Tokenizer is required for caption generation"
        )

        # ------------------------------------------------------------------
        # 1. Build the *input* token matrix for teacher forcing
        # ------------------------------------------------------------------
        max_len = self.max_caption_length
        pad_id = self.pad_token_id
        eos_id = self.end_token_id

        # (B, N, L) filled with PAD
        input_ids = torch.full((B, N, max_len), pad_id, dtype=torch.long, device=device)

        for b in range(B):
            gt_objs = ground_truth_objects[b]
            num_gt = len(gt_objs)
            # Limit to N queries; extra GT objects will be handled by Hungarian
            # matching in the loss function.
            for q in range(N):
                if q < num_gt:
                    # Encode description – include special tokens so that the
                    # teacher-forced sequence matches the target used in
                    # DetectionLoss.
                    tokens = self.tokenizer(
                        gt_objs[q]["desc"],
                        padding="max_length",
                        truncation=True,
                        max_length=max_len,
                        return_tensors="pt",
                    ).input_ids.to(device)
                    input_ids[b, q] = tokens[0]
                else:
                    # Unmatched query → only EOS token (position 0).  Targets
                    # in DetectionLoss will also be EOS for these queries.
                    input_ids[b, q, 0] = eos_id

        # ------------------------------------------------------------------
        # 2. Token + position embeddings  →  caption features  (B*N, L, D)
        # ------------------------------------------------------------------
        token_embeds = self.token_embedding(input_ids.view(-1, max_len))  # (B*N,L,D)
        if token_embeds.dtype != dtype:
            token_embeds = token_embeds.to(dtype)

        # Rotary-style sinusoidal positional encodings (sin,cos)
        position_ids = torch.arange(max_len, device=device).float()  # (L,)
        # Compute inverse frequencies for half dim
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, D, 2, device=device).float() / D)
        )  # (D/2,)
        # Outer product: (L, D/2)
        sinusoid_inp = torch.einsum("i,j->ij", position_ids, inv_freq)
        sin_part = sinusoid_inp.sin()  # (L, D/2)
        cos_part = sinusoid_inp.cos()  # (L, D/2)
        # Interleave sin and cos to shape (L, D)
        pos_emb = torch.zeros((max_len, D), dtype=dtype, device=device)
        pos_emb[:, 0::2] = sin_part
        pos_emb[:, 1::2] = cos_part
        caption_features = token_embeds + pos_emb  # (B*N, L, D)

        # ------------------------------------------------------------------
        # 3. Transformer decoding with cross-attention to the object feature
        # ------------------------------------------------------------------
        causal_mask = torch.triu(
            torch.ones(max_len, max_len, device=device, dtype=torch.bool), 1
        )

        # Flatten object_features to (B*N, 1, D) memory for cross-attention
        memory = object_features.reshape(B * N, D).unsqueeze(1)

        decoded = self.caption_decoder(
            tgt=caption_features,
            memory=memory,
            tgt_mask=causal_mask,
        )  # (B*N, L, D)

        # ------------------------------------------------------------------
        # 4. Project to vocabulary and reshape back to (B, N, L, V)
        # ------------------------------------------------------------------
        logits = self.caption_head(decoded)  # (B*N, L, vocab)
        logits = logits.reshape(B, N, max_len, -1)

        # Clamp extreme values to keep the loss finite
        logits = torch.clamp(logits, -20.0, 20.0)
        return logits

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
            # Create logits where pad token is highly likely, but avoid -inf
            # to prevent numerical instability if the target is not the pad token.
            pad_logits = torch.zeros_like(all_logits[0])
            pad_logits[..., self.pad_token_id] = 1e9  # A large value
            all_logits.append(pad_logits)

        # Stack to get (B, N, max_len, vocab_size)
        caption_logits = torch.stack(all_logits, dim=2)

        # Prevent extreme logit values that cause loss explosion
        caption_logits = torch.clamp(caption_logits, min=-20.0, max=20.0)

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

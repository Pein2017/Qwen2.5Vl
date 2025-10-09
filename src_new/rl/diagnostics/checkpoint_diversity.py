"""
Checkpoint Diversity Comparison Tool

Feature: 004-grpo-post-training
Constitution: v4.1.1

Compares generation diversity between different SFT checkpoints.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from src_new.processing.conversation_builder import ConversationBuilder


@dataclass
class DiversityMetrics:
    """Diversity metrics for a checkpoint."""
    
    checkpoint_name: str
    num_prompts: int
    k_completions: int
    
    # Token-level diversity
    unique_completion_ratio: float  # Ratio of unique completions
    avg_edit_distance: float  # Average Levenshtein distance
    avg_token_overlap: float  # Average token overlap (1.0 = identical)
    
    # Length diversity
    mean_completion_length: float
    std_completion_length: float
    length_range: Tuple[int, int]
    
    # Vocabulary diversity
    unique_tokens_per_prompt: float  # Avg unique tokens across K completions
    vocab_size: int  # Total unique tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON export."""
        return {
            "checkpoint_name": self.checkpoint_name,
            "num_prompts": self.num_prompts,
            "k_completions": self.k_completions,
            "unique_completion_ratio": self.unique_completion_ratio,
            "avg_edit_distance": self.avg_edit_distance,
            "avg_token_overlap": self.avg_token_overlap,
            "mean_completion_length": self.mean_completion_length,
            "std_completion_length": self.std_completion_length,
            "length_range": list(self.length_range),
            "unique_tokens_per_prompt": self.unique_tokens_per_prompt,
            "vocab_size": self.vocab_size,
        }


def levenshtein_distance(s1: List[int], s2: List[int]) -> int:
    """Compute Levenshtein distance between two token sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def compute_token_overlap(tokens1: List[int], tokens2: List[int]) -> float:
    """Compute token overlap ratio (Jaccard similarity)."""
    if not tokens1 and not tokens2:
        return 1.0
    
    set1 = set(tokens1)
    set2 = set(tokens2)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def analyze_checkpoint_diversity(
    checkpoint_path: str,
    prompts: List[str],
    k_completions: int,
    temperature: float,
    max_new_tokens: int,
    device: str = "cuda",
) -> DiversityMetrics:
    """
    Analyze generation diversity for a single checkpoint.
    
    Args:
        checkpoint_path: Path to SFT checkpoint
        prompts: List of prompt strings
        k_completions: Number of completions per prompt
        temperature: Generation temperature
        max_new_tokens: Max tokens to generate
        device: Device to run on
        
    Returns:
        DiversityMetrics for this checkpoint
    """
    from transformers import Qwen2VLForConditionalGeneration
    
    # Load model and tokenizer
    print(f"Loading checkpoint: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    all_completions = []
    all_tokens = []
    unique_counts = []
    edit_distances = []
    token_overlaps = []
    unique_tokens_per_prompt = []
    
    with torch.no_grad():
        for prompt in prompts:
            # Generate K completions for this prompt
            completions_for_prompt = []
            tokens_for_prompt = []
            
            for k in range(k_completions):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.05,
                )
                
                # Decode only new tokens
                new_tokens = outputs[0, inputs.input_ids.size(1):]
                completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                completions_for_prompt.append(completion)
                tokens_for_prompt.append(new_tokens.tolist())
                all_tokens.append(new_tokens.tolist())
            
            # Compute diversity metrics for this prompt
            unique_completions = len(set(completions_for_prompt))
            unique_counts.append(unique_completions / k_completions)
            
            # Pairwise edit distances
            for i in range(k_completions):
                for j in range(i + 1, k_completions):
                    dist = levenshtein_distance(
                        tokens_for_prompt[i],
                        tokens_for_prompt[j]
                    )
                    edit_distances.append(dist)
                    
                    overlap = compute_token_overlap(
                        tokens_for_prompt[i],
                        tokens_for_prompt[j]
                    )
                    token_overlaps.append(overlap)
            
            # Unique tokens for this prompt
            all_tokens_prompt = set()
            for tokens in tokens_for_prompt:
                all_tokens_prompt.update(tokens)
            unique_tokens_per_prompt.append(len(all_tokens_prompt))
    
    # Aggregate metrics
    lengths = [len(tokens) for tokens in all_tokens]
    all_vocab = set()
    for tokens in all_tokens:
        all_vocab.update(tokens)
    
    metrics = DiversityMetrics(
        checkpoint_name=Path(checkpoint_path).name,
        num_prompts=len(prompts),
        k_completions=k_completions,
        unique_completion_ratio=sum(unique_counts) / len(unique_counts),
        avg_edit_distance=sum(edit_distances) / len(edit_distances) if edit_distances else 0.0,
        avg_token_overlap=sum(token_overlaps) / len(token_overlaps) if token_overlaps else 0.0,
        mean_completion_length=sum(lengths) / len(lengths),
        std_completion_length=torch.tensor(lengths, dtype=torch.float32).std().item(),
        length_range=(min(lengths), max(lengths)),
        unique_tokens_per_prompt=sum(unique_tokens_per_prompt) / len(unique_tokens_per_prompt),
        vocab_size=len(all_vocab),
    )
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return metrics


def compare_checkpoints(
    checkpoint_paths: List[str],
    prompts: List[str],
    k_completions: int = 8,
    temperatures: List[float] = [0.7, 1.0, 1.3],
    max_new_tokens: int = 512,
    output_dir: str = "outputs/checkpoint_diversity",
) -> Dict[str, List[DiversityMetrics]]:
    """
    Compare diversity across multiple checkpoints and temperatures.
    
    Returns:
        Dict mapping checkpoint names to list of DiversityMetrics (one per temperature)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for checkpoint_path in checkpoint_paths:
        checkpoint_name = Path(checkpoint_path).name
        results[checkpoint_name] = []
        
        for temp in temperatures:
            print(f"\nAnalyzing {checkpoint_name} at temperature {temp}")
            
            metrics = analyze_checkpoint_diversity(
                checkpoint_path=checkpoint_path,
                prompts=prompts,
                k_completions=k_completions,
                temperature=temp,
                max_new_tokens=max_new_tokens,
            )
            
            results[checkpoint_name].append(metrics)
            
            # Save individual result
            result_file = output_path / f"{checkpoint_name}_temp{temp}.json"
            with open(result_file, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)
            
            print(f"  Unique ratio: {metrics.unique_completion_ratio:.2%}")
            print(f"  Avg edit distance: {metrics.avg_edit_distance:.1f}")
            print(f"  Avg token overlap: {metrics.avg_token_overlap:.2%}")
    
    # Save comparison summary
    summary_file = output_path / "comparison_summary.json"
    summary = {}
    for ckpt, metrics_list in results.items():
        summary[ckpt] = [m.to_dict() for m in metrics_list]
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return results

"""Compare EOS generation behavior across SFT checkpoints using DetectionModel"""

import sys


sys.path.insert(0, "/data3/Qwen2.5-VL-main")

import json
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from src_new.models.wrapper import DetectionModel


@dataclass
class SimpleConfig:
    """Minimal config for DetectionModel"""

    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    merge_size: int = 2


checkpoints = [
    (
        "Phase2",
        "outputs/7B-all_tokens/phase_2/9-21-phase_2-last_blocks_6-all_tokens/best-200-eval_loss0.7006",
    ),
    (
        "Phase3-base",
        "outputs/7B-all_tokens/phase_3/9-22-phase_3-all_tokens-lower_grounding_weight-last_vision_4/best-800-eval_loss0.2135",
    ),
    (
        "Phase3-text",
        "outputs/7B-all_tokens/phase_3/9-22-phase_3-all_tokens-lower_grounding_weight-last_vision_4-with_text_only/best-800-eval_loss0.2137",
    ),
    (
        "Phase3-resume",
        "outputs/7B-all_tokens/phase_3/9-23-phase_3-all_tokens-lower_grounding_weight-last_vision_4-with_text_only-resume/best-800-eval_loss0.1004",
    ),
]

# Simple text prompt
test_prompt = """<|im_start|>user
Describe the objects.<|im_end|>
<|im_start|>assistant
"""

print("=" * 90)
print(" " * 25 + "CHECKPOINT EOS GENERATION COMPARISON")
print("=" * 90)
print(f"\nTest: {test_prompt.strip()[:50]}...")
print()

results = []

for ckpt_name, checkpoint in checkpoints:
    print(f"\n{'=' * 90}")
    print(f"{ckpt_name}")
    print(f"{'=' * 90}")

    try:
        # Load tokenizer
        print("  [1/2] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        eos_token_id = tokenizer.eos_token_id
        print(f"         EOS: '{tokenizer.eos_token}' (ID: {eos_token_id})")

        # Load model using DetectionModel wrapper
        print("  [2/2] Loading model...")
        config = SimpleConfig()
        model = DetectionModel.from_pretrained(
            checkpoint,
            config=config,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        model.eval()

        # Tokenize
        inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda:0")
        prompt_len = inputs["input_ids"].shape[1]

        # Test with different max_new_tokens
        test_configs = [512, 1024, 2048]

        checkpoint_results = {"checkpoint": ckpt_name, "tests": []}

        print("         Testing generation lengths...")
        for max_tokens in test_configs:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    eos_token_id=eos_token_id,
                    do_sample=False,
                )

            completion_ids = outputs[0][prompt_len:]
            completion_len = len(completion_ids)
            hit_cap = completion_len >= max_tokens

            # Check for EOS token ID
            eos_in_ids = eos_token_id in completion_ids.tolist()
            eos_pos = completion_ids.tolist().index(eos_token_id) if eos_in_ids else -1

            test_result = {
                "max_tokens": max_tokens,
                "completion_len": completion_len,
                "has_eos": eos_in_ids,
                "eos_position": eos_pos,
                "hit_cap": hit_cap,
            }
            checkpoint_results["tests"].append(test_result)

            # Format output
            eos_marker = "✅" if eos_in_ids else "❌"
            cap_marker = " CAP" if hit_cap else "    "
            eos_info = f"@{eos_pos:4d}" if eos_in_ids else "    "

            print(
                f"         {eos_marker} {max_tokens:4d}tok → {completion_len:4d}tok {eos_info} {cap_marker}"
            )

        results.append(checkpoint_results)

        # Cleanup
        del model
        torch.cuda.empty_cache()
        print("  ✓ Complete")

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        results.append({"checkpoint": ckpt_name, "error": str(e)})

# Final Summary
print("\n" + "=" * 90)
print(" " * 35 + "SUMMARY")
print("=" * 90)

best_checkpoint = None
best_score = 0

for result in results:
    if "error" in result:
        print(f"\n❌ {result['checkpoint']}: FAILED")
        continue

    eos_count = sum(1 for test in result["tests"] if test["has_eos"])
    natural_eos = sum(
        1 for test in result["tests"] if test["has_eos"] and not test["hit_cap"]
    )

    # Show results
    status = "✅" if eos_count > 0 else "❌"
    print(f"\n{status} {result['checkpoint']}: {eos_count}/3 tests generated EOS")
    if eos_count > 0:
        print(f"   Natural EOS (no cap): {natural_eos}/3")
        # Show at which lengths EOS appears
        for test in result["tests"]:
            if test["has_eos"]:
                cap_note = " (hit cap)" if test["hit_cap"] else ""
                print(
                    f"   - {test['max_tokens']}tok: EOS at position {test['eos_position']}{cap_note}"
                )

    # Score: prefer natural EOS over hitting cap
    score = natural_eos * 10 + eos_count * 5
    if score > best_score:
        best_score = score
        best_checkpoint = result["checkpoint"]

if best_checkpoint:
    print(f"\n🏆 BEST CHECKPOINT: {best_checkpoint}")
else:
    print(f"\n⚠️  NO checkpoints generated EOS - all need longer max_new_tokens!")

# Save
with open("checkpoint_eos_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Detailed results: checkpoint_eos_comparison.json")
print("=" * 90)

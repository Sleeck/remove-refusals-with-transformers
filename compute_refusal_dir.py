import os
import random
from pathlib import Path

import torch
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SOURCE_MODEL_ID = "Qwen/Qwen3.5-9"
EXPORT_ROOT = Path("exports")

HF_WRITE_TOKEN = os.getenv("HF_WRITE_TOKEN")
HF_DESTINATION_REPO_ID = os.getenv("HF_DESTINATION_REPO_ID")
HF_PRIVATE_REPO = os.getenv("HF_PRIVATE_REPO", "false").lower() in {"1", "true", "yes"}

# settings:
INSTRUCTIONS_COUNT = 32
LAYER_POSITION = 0.6
TOKEN_POSITION = -1


def validate_required_env_vars():
    missing = []
    if not HF_WRITE_TOKEN:
        missing.append("HF_WRITE_TOKEN")
    if not HF_DESTINATION_REPO_ID:
        missing.append("HF_DESTINATION_REPO_ID")

    if missing:
        raise ValueError(
            "Missing required environment variable(s): "
            f"{', '.join(missing)}. "
            "Example: HF_WRITE_TOKEN=hf_xxx HF_DESTINATION_REPO_ID=username/model python3 compute_refusal_dir.py"
        )


@torch.inference_mode()
def generate(model, bar, toks):
    bar.update(n=1)
    model_inputs = toks.to(model.device)
    if hasattr(model_inputs, "items"):
        return model.generate(
            **dict(model_inputs),
            use_cache=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    return model.generate(
        model_inputs,
        use_cache=False,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_hidden_states=True,
    )


def main():
    validate_required_env_vars()

    print(f"Downloading source model: {SOURCE_MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL_ID,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="cuda",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_ID, trust_remote_code=True)

    layer_idx = int(len(model.model.layers) * LAYER_POSITION)

    print(f"Instruction count: {INSTRUCTIONS_COUNT}")
    print(f"Layer index: {layer_idx}")

    with open("harmful.txt", "r") as f:
        harmful = f.readlines()

    with open("harmless.txt", "r") as f:
        harmless = f.readlines()

    harmful_instructions = random.sample(harmful, INSTRUCTIONS_COUNT)
    harmless_instructions = random.sample(harmless, INSTRUCTIONS_COUNT)

    harmful_toks = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": insn}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        for insn in harmful_instructions
    ]
    harmless_toks = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": insn}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        for insn in harmless_instructions
    ]

    bar = tqdm(total=INSTRUCTIONS_COUNT * 2)
    harmful_outputs = [generate(model, bar, toks) for toks in harmful_toks]
    harmless_outputs = [generate(model, bar, toks) for toks in harmless_toks]
    bar.close()

    harmful_hidden = [output.hidden_states[0][layer_idx][:, TOKEN_POSITION, :] for output in harmful_outputs]
    harmless_hidden = [output.hidden_states[0][layer_idx][:, TOKEN_POSITION, :] for output in harmless_outputs]

    harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
    harmless_mean = torch.stack(harmless_hidden).mean(dim=0)

    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()

    destination_slug = HF_DESTINATION_REPO_ID.replace("/", "__")
    export_dir = EXPORT_ROOT / destination_slug
    export_dir.mkdir(parents=True, exist_ok=True)

    refusal_path = export_dir / "refusal_dir.pt"
    torch.save(refusal_dir.cpu(), refusal_path)
    model.save_pretrained(export_dir, safe_serialization=True)
    tokenizer.save_pretrained(export_dir)

    print(f"Refusal direction saved to: {refusal_path}")
    print(f"Prepared upload folder: {export_dir}")

    print(f"Uploading artifacts to Hugging Face repo: {HF_DESTINATION_REPO_ID}")
    api = HfApi(token=HF_WRITE_TOKEN)
    api.create_repo(
        repo_id=HF_DESTINATION_REPO_ID,
        token=HF_WRITE_TOKEN,
        repo_type="model",
        private=HF_PRIVATE_REPO,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(export_dir),
        repo_id=HF_DESTINATION_REPO_ID,
        token=HF_WRITE_TOKEN,
        repo_type="model",
    )

    print(f"Upload complete: https://huggingface.co/{HF_DESTINATION_REPO_ID}")


if __name__ == "__main__":
    main()

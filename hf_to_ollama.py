import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

WORKDIR = Path("ollama_build")

HF_SOURCE_MODEL_ID = os.getenv("HF_SOURCE_MODEL_ID")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")
GGUF_QUANT = os.getenv("GGUF_QUANT", "Q4_K_M")
LLAMA_CPP_DIR = Path(os.getenv("LLAMA_CPP_DIR", "./llama.cpp"))


def run(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def validate_env():
    missing = []
    if not HF_SOURCE_MODEL_ID:
        missing.append("HF_SOURCE_MODEL_ID")
    if not OLLAMA_MODEL_NAME:
        missing.append("OLLAMA_MODEL_NAME")

    if missing:
        raise ValueError(
            "Missing required environment variable(s): "
            f"{', '.join(missing)}. "
            "Example: HF_SOURCE_MODEL_ID=Qwen/Qwen2.5-3B-Instruct OLLAMA_MODEL_NAME=my-qwen python3 hf_to_ollama.py"
        )


def validate_tools():
    if shutil.which("ollama") is None:
        raise RuntimeError("`ollama` not found in PATH.")

    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"

    if not convert_script.exists():
        raise RuntimeError(f"Missing converter: {convert_script}")
    if not quantize_bin.exists():
        raise RuntimeError(
            f"Missing quantizer binary: {quantize_bin}. Build llama.cpp first, e.g. `cmake -B build && cmake --build build -j`."
        )

    return convert_script, quantize_bin


def write_modelfile(modelfile_path, gguf_path):
    modelfile_path.write_text(
        "\n".join(
            [
                f"FROM {gguf_path.resolve()}",
                "TEMPLATE \"\"\"{{ .Prompt }}\"\"\"",
                "PARAMETER temperature 0.7",
                "",
            ]
        )
    )


def main():
    validate_env()
    convert_script, quantize_bin = validate_tools()

    WORKDIR.mkdir(parents=True, exist_ok=True)
    source_dir = WORKDIR / HF_SOURCE_MODEL_ID.replace("/", "__")
    source_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model from Hugging Face: {HF_SOURCE_MODEL_ID}")
    snapshot_download(repo_id=HF_SOURCE_MODEL_ID, local_dir=str(source_dir), local_dir_use_symlinks=False)

    f16_path = WORKDIR / f"{OLLAMA_MODEL_NAME}.f16.gguf"
    quant_path = WORKDIR / f"{OLLAMA_MODEL_NAME}.{GGUF_QUANT.lower()}.gguf"
    modelfile_path = WORKDIR / f"Modelfile.{OLLAMA_MODEL_NAME}"

    run(["python3", str(convert_script), str(source_dir), "--outfile", str(f16_path), "--outtype", "f16"])
    run([str(quantize_bin), str(f16_path), str(quant_path), GGUF_QUANT])

    write_modelfile(modelfile_path, quant_path)

    run(["ollama", "create", OLLAMA_MODEL_NAME, "-f", str(modelfile_path)])
    print(f"Done. You can now run: ollama run {OLLAMA_MODEL_NAME}")


if __name__ == "__main__":
    main()

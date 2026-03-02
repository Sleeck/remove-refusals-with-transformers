# Removing refusals with HF Transformers

This is a crude, proof-of-concept implementation to remove refusals from an LLM model without using TransformerLens. This means, that this supports every model that HF Transformers supports*.

The code was tested on a RTX 2060 6GB, thus mostly <3B models have been tested, but the code has been tested to work with bigger models as well.

*While most models are compatible, some models are not. Mainly because of custom model implementations. Some Qwen implementations for example don't work. Because `model.model.layers` can't be used for getting layers. They call the variables so that, `model.transformer.h` must be used, if I'm not mistaken.

## Usage
1. Set model and quantization in compute_refusal_dir.py and inference.py (Quantization can apparently be mixed)
2. Run compute_refusal_dir.py (Some settings in that file may be changed depending on your use-case)
3. Run inference.py and ask the model how to build an army of rabbits, that will overthrow your local government one day, by stealing all the carrots.

## Credits
- [Harmful instructions](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv)
- [Harmless instructions](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- [Technique](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)

## Upload to Hugging Face (one-shot pipeline)
`compute_refusal_dir.py` now does everything in one run:
1. download source model (`SOURCE_MODEL_ID` in the script),
2. compute `refusal_dir.pt`,
3. export full artifacts locally,
4. upload the folder to Hugging Face.

### Required environment variables
Set these **before** running the script:

```bash
export HF_WRITE_TOKEN=hf_xxx
export HF_DESTINATION_REPO_ID=your-username/your-model-repo
```

Optional:

```bash
export HF_PRIVATE_REPO=true   # create destination repo as private
```

### Run
```bash
python3 compute_refusal_dir.py
```

### Output
Local artifacts are stored in:
- `exports/<destination_repo_id_with_double_underscores>/`

This folder contains:
- model files (`config.json`, `model.safetensors`, etc.)
- tokenizer files
- `refusal_dir.pt`

If required variables are missing, the script fails immediately with an explicit error message.

## One-shot Ollama depuis Hugging Face
Le script `hf_to_ollama.py` fait le pipeline complet :
1. téléchargement du modèle HF,
2. conversion en GGUF F16 (via `llama.cpp`),
3. quantization GGUF,
4. création automatique du modèle Ollama.

### Variables obligatoires
```bash
export HF_SOURCE_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
export OLLAMA_MODEL_NAME=qwen25-local
```

### Variables optionnelles
```bash
export GGUF_QUANT=Q4_K_M
export LLAMA_CPP_DIR=/chemin/vers/llama.cpp
```

### Pré-requis
- `ollama` installé et dans le `PATH`
- `llama.cpp` cloné et compilé avec `build/bin/llama-quantize`

Exemple build rapide:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build -j
```

### Exécution
```bash
python3 hf_to_ollama.py
ollama run "$OLLAMA_MODEL_NAME"
```

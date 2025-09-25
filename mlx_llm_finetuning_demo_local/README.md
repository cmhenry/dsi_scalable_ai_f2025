---
title: "mlx_llm_finetuning_demo_local"
author: "Maël Kubli"
date: "2025-09-24"
---

Runnable demo of finetuning an LLM on Apple Silicon using the MLX framework from Apple.  
This script showcases how you can finetune an open-source LLM locally on your personal Apple workstation (Apple Silicon only).

## Repository Structure
```
mlx_llm_finetuning_demo_local
├── README.md
├── adapters
│   └── testing
│       ├── 0000100_adapters.safetensors
│       ├── 0000200_adapters.safetensors
│       ├── adapter_config.json
│       └── adapters.safetensors
├── data
│   └── testing
│       ├── test.json
│       └── train.json
├── llama.cpp-master.zip
├── model
└── src
    ├── MLX LM LoRA Fine Tune LLaMA 3 8b.ipynb
    ├── MLX-LM_to_gguf.ipynb
    └── MLX-Training-Data-Preparation.R
```


## Instructions
### 1) Prepare Training Data
Open the `MLX-Training-Data-Preparation.R` script and adjust it according to your needs. Prepare the `test.json` and `train.json` files for the finetuning of your preferred open-source LLM.

### 2) Download Preferred LLM
Make sure you have access to the open-source LLM on Hugging Face. If not, request access to the preferred model.  
For example, for LLaMA 3.1 go to:  
<https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> and request access.

### 3) Finetune LLM with MLX
Open the Jupyter Notebook titled `MLX LM LoRA Fine Tune LLaMA 3 8b.ipynb` and apply the necessary changes to load your training and test datasets, as well as to adapt the prompts to your needs. Otherwise, it will run on the template data and instructions.

### 4) Transform Model to GGUF for llama.cpp (optional)
Run the `MLX-LM_to_gguf.ipynb` notebook to transform the output into a `.gguf` model file for use with `llama.cpp` or `ollama`.

For **ollama**, after conversion, put the `.gguf` model into the `ollama-models` folder and then open a terminal to run the following:
 
```
# ~/ollama-models/your_model_name/Modelfile
FROM ./your_model_name.gguf

TEMPLATE """<s>[INST] {{ .Prompt }} [/INST]"""

PARAMETER temperature 0.0
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

SYSTEM """You are a helpful AI assistant. Respond clearly and concisely to user questions."""
``` 

After which you navigate to your model directory and create the ollama model:
```
cd ~/ollama-models/your_model_name

# Import model with custom name
ollama create your_model_name -f Modelfile
```

For more information on importing custom models into ollama see: https://markaicode.com/import-gguf-models-ollama-guide/
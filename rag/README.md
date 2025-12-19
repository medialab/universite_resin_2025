# Create a RAG environment

Copy the following instructions in a terminal:
```
mkdir -p ~/.local/share/jupyter/kernels/rag
cp /pbs/throng/training/universite-hiver/rag-envs/rag2/kernel.json ~/.local/share/jupyter/kernels/rag
```
Then log out of your notebook, and log back in.


# Launch an Ollama backend
Copy the following instructions into a separate terminal:
```
module purge
module load ollama
export HUGGINGFACE_HUB_CACHE=/pbs/throng/training/universite-hiver/cache/huggingface
export OLLAMA_MODELS=/pbs/throng/training/universite-hiver/cache/ollama/models
export OLLAMA_HOST=127.0.0.1:65383
ollama serve &
```
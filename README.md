# Gemma 4 Playground

**This repository is for testing purposes.**

## Python

Gemma 4 original version

<https://huggingface.co/google/gemma-4-E2B-it>

```sh
cd python/
uv sync --frozen
uv run main.py
```

## WebGPU

ONNX version

<https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX>

```sh
cd webgpu-onnx/
pnpm install --frozen-lockfile
pnpm run dev
```

Open Chrome's DevTools to read outputs.

## Update dependencies

```sh
cd python/
uv sync --upgrade
cd ../webgpu-onnx/
pnpm update
git add -u
git commit -m 'chore: update dependencies
```

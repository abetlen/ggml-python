# Replit Code Completion Server

This example is a local-first Github Copilot drop-in replacement using the replit-code-v1-3b model written entirely in ggml-python.

For best performance (likely still slower than copilot) please run with CUDA, OpenCL, or Metal support.


## Installation

```bash
# Clone the repo
git clone https://github.com/abetlen/ggml-python.git
cd ggml-python/examples/replit
# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

## Model Weights

You can download the quantized model weights from [here](https://huggingface.co/abetlen/replit-code-v1-3b-ggml)

## Running the Server

```bash
# Start the server
MODEL=/path/to/model uvicorn server:app --reload
```

## Editor Setup

### VSCode

Add the following to your `settings.json`:

```json
{
    "github.copilot.advanced": {
        "debug.testOverrideProxyUrl": "http://localhost:8000",
        "debug.overrideProxyUrl": "http://localhost:8000"
    }
}
```

### Vim / Neovim

Add the following to your vimrc or init.vim:

```
let g:copilot_proxy = 'localhost:8000'
let g:copilot_strict_ssl = 0
```
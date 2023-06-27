# CLIP Example

# Setup

Create a virtual environment and install requirements.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Convert the original CLIP model to GGML format.

```bash
python convert-pt-to-ggml.py ViT-B/32 ./models
```

The other CLIP vision transformers should work, but have not been tested. Namely:

- ViT-B/16
- ViT-L/14
- ViT-L/14@336px

# Run the example

```bash
python example.py
```

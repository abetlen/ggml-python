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

# Usage

```python
# This implements the same example as the original project: https://github.com/openai/CLIP#usage
from model import ClipModel
from scipy.special import softmax
from PIL import Image
from utils import tokenize, transform


preprocess = transform(224)
# Example image: https://github.com/openai/CLIP/blob/main/CLIP.png
image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
text = tokenize(["a diagram", "a dog", "a cat"])

# Initialize Model
model_file = "models/ViT-B-32.ggml"
model = ClipModel.init_from_file(model_file, n_threads=1)

# Features are computed one at a time, batching not supported yet
text_features = model.encode_text(text)

# Only single image supported in ggml right now
image_features = model.encode_image(image)

logits_per_image, logits_per_text = model(image, text)

probs = softmax(logits_per_image)

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
```

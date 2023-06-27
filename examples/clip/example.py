# This example implements the same example as the CLIP example from https://github.com/openai/CLIP#usage

from model import ClipModel
from scipy.special import softmax
import clip
from PIL import Image
from simple_tokenizer import tokenize
from transform import transform

preprocess = transform(224)

model_file = "models/ViT-B-32.ggml"
model = ClipModel.init_from_file(model_file, n_threads=1)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
text = tokenize(["a diagram", "a dog", "a cat"])

# # Features are computed one at a time, batching not supported yet
text_features = model.encode_text(text)

# # Only single image supported in ggml right now
image_features = model.encode_image(image)

logits_per_image, logits_per_text = model(image, text)

probs = softmax(logits_per_image)

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

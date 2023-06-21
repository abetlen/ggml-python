import IPython
from model import ClipModel
from CLIP import clip
from PIL import Image

preprocess = clip.clip._transform(224)

model_file = "models/ViT-B-32.ggml"
model = ClipModel.init_from_file(model_file, n_threads=1)

image = preprocess(Image.open("CLIP/CLIP.png")).unsqueeze(0)
text = clip.tokenize(["a diagram", "a dog", "a cat"])

# Only single image supported in ggml right now
image_features = model.encode_image(image)

IPython.embed()
text_features = model.encode_text(text)


logits_per_image, logits_per_text = model(image, text)
probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

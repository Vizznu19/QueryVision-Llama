import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Sentence embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# 🔴 CHANGE THIS
image_folder = "test_images"

# 🔴 MODIFY WITH YOUR REAL GROUND TRUTH
ground_truth = {
    "img1.jpg": "a person standing near a car",
    "img2.jpg": "a man riding a motorcycle"
}

def compute_similarity(pred, true):
    emb1 = embed_model.encode([pred])
    emb2 = embed_model.encode([true])
    return cosine_similarity(emb1, emb2)[0][0]

results = {}

# ================= BLIP =================
print("\nRunning BLIP...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

scores = []
for img_name in ground_truth:
    image = Image.open(os.path.join(image_folder, img_name)).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs)

    caption = processor.decode(output[0], skip_special_tokens=True)
    score = compute_similarity(caption, ground_truth[img_name])
    scores.append(score)

results["BLIP"] = np.mean(scores)

del model
torch.cuda.empty_cache()

# ================= CLIP =================
print("\nRunning CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

scores = []
for img_name in ground_truth:
    image = Image.open(os.path.join(image_folder, img_name)).convert("RGB")

    inputs = clip_processor(
        text=[ground_truth[img_name]],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)

    score = outputs.logits_per_image.softmax(dim=1)[0][0].item()
    scores.append(score)

results["CLIP"] = np.mean(scores)

del clip_model
torch.cuda.empty_cache()

print("\nFinal Results:", results)

# ======== Plot Graph ========
plt.figure()
plt.bar(results.keys(), results.values())
plt.xlabel("Model")
plt.ylabel("Average Similarity Score")
plt.title("VLM Comparison")
plt.savefig("vlm_comparison.png", dpi=300)
plt.show()
import requests
import torch
import numpy as np
import faiss
import clip
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import json
import gc

session = requests.Session()

# ---------------- CNN MODEL ----------------
import torch.nn as nn
import torch.nn.functional as F

class WatchCNN(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return F.normalize(self.fc(x), dim=1)


# ---------------- SERVICE ----------------
class ImageSimilarityService:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

        self.cnn_model = WatchCNN().to(self.device)
        self.cnn_model.eval()

        self.index = None
        self.metadata = []

    # ---------------- UTILS ----------------
    def download_image(self, url):
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code != 200:
                return None
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except:
            return None

    def center_crop(self, img):
        w, h = img.size
        return img.crop((w//4, h//4, w*3//4, h*3//4))

    def zoom_crop(self, img):
        w, h = img.size
        return img.crop((w//3, h//3, w*2//3, h*2//3))

    # ---------------- EMBEDDING ----------------
    def get_clip_embedding(self, img):
        img = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(img)
        emb = emb.cpu().numpy()[0]
        return emb / np.linalg.norm(emb)

    def get_cnn_embedding(self, img):
        img = img.resize((224, 224))
        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.cnn_model(img)

        return emb.cpu().numpy()[0]

    # ---------------- MULTI-CROP EMBEDDING ----------------
    def get_final_embedding(self, img):

        center = self.center_crop(img)
        zoom = self.zoom_crop(img)

        # CLIP embeddings
        clip_full = self.get_clip_embedding(img)
        clip_center = self.get_clip_embedding(center)
        clip_zoom = self.get_clip_embedding(zoom)

        clip = (clip_full + clip_center + clip_zoom) / 3

        # CNN embeddings
        cnn_full = self.get_cnn_embedding(img)
        cnn_center = self.get_cnn_embedding(center)

        cnn = (cnn_full + cnn_center) / 2

        # combine
        emb = np.concatenate([clip * 0.7, cnn * 0.3])

        # 🔥 VERY IMPORTANT (fix overflow bug)
        emb = emb / np.linalg.norm(emb)

        return emb.astype("float32")

    # ---------------- BUILD ----------------
    def build(self, products):

        embeddings = []
        metadata = []

        print("Building index...")

        def safe_download(p):
            try:
                resp = requests.get(p["image"], timeout=10)
                if resp.status_code != 200:
                    return None, p
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                return img, p
            except:
                return None, p

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(safe_download, products))

        for i, (img, p) in enumerate(results):

            if img is None:
                continue

            try:
                emb = self.get_final_embedding(img)
                embeddings.append(emb)
                metadata.append(p)
            except:
                continue

            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(products)}")

        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.index.add(embeddings)

        self.metadata = metadata

        faiss.write_index(self.index, "index3.faiss")
        json.dump(self.metadata, open("meta3.json", "w"))

        print("BUILD DONE")

    # ---------------- LOAD ----------------
    def load(self):
        self.index = faiss.read_index("index3.faiss")
        self.metadata = json.load(open("meta3.json"))

    # ---------------- SEARCH ----------------
    def search(self, image_url, top_k=5):

        img = self.download_image(image_url)
        if img is None:
            return {"success": False}

        query = self.get_final_embedding(img)

        D, I = self.index.search(query.reshape(1, -1), 20)

        results = []

        for score, idx in zip(D[0], I[0]):
            if idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]

            results.append({
                "id": meta["id"],
                "product_id": meta["product_id"],
                "image": meta["image"],
                "score": float(score)
            })

        if not results:
            return {"success": False}

        # sort properly
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return {
            "success": True,
            "best_match": results[0],
            "similar": results
        }


# INIT
service = ImageSimilarityService()
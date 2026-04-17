import requests
import torch
import numpy as np
import faiss
from PIL import Image
from io import BytesIO
import json
import imagehash
import cv2

import torch.nn as nn
import torch.nn.functional as F

# ---------------- CNN MODEL ----------------
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

        # 🔥 CLIP (stable)
        import clip
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

        self.cnn_model = WatchCNN().to(self.device)
        self.cnn_model.eval()

        self.index = None
        self.metadata = []

    # ---------------- DOWNLOAD ----------------
    def download_image(self, url):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return None
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except:
            return None

    # ---------------- NORMALIZE ----------------
    def normalize_image(self, img):
        img = img.resize((512, 512))
        img = np.array(img)

        img = np.clip(img * 1.1, 0, 255).astype("uint8")
        img = cv2.GaussianBlur(img, (3, 3), 0)

        return Image.fromarray(img)

    # ---------------- CROPS ----------------
    def center_crop(self, img):
        w, h = img.size
        return img.crop((w//4, h//4, w*3//4, h*3//4))

    def get_dial_crop(self, img):
        w, h = img.size
        cx, cy = w // 2, h // 2
        r = min(w, h) // 3
        return img.crop((cx - r, cy - r, cx + r, cy + r))

    # ---------------- AUGMENT ----------------
    def generate_variants(self, img):
        variants = [img]

        for angle in [-20, -10, 10, 20]:
            variants.append(img.rotate(angle))

        w, h = img.size
        variants.append(img.crop((w*0.1, h*0.1, w*0.9, h*0.9)))

        return variants

    # ---------------- COLOR ----------------
    def get_color(self, img):
        img = img.resize((100, 100))
        arr = np.array(img).reshape(-1, 3)
        return np.mean(arr, axis=0)

    def color_sim(self, c1, c2):
        return 1 - np.linalg.norm(c1 - c2) / 255.0

    # ---------------- EMBEDDINGS ----------------
    def get_clip_embedding(self, img):
        img = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0].astype("float32")

    def get_cnn_embedding(self, img):
        img = img.resize((224, 224))
        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.cnn_model(img)

        return emb.cpu().numpy()[0]

    def get_final_embedding(self, img):
        img = self.normalize_image(img)
        dial = self.get_dial_crop(img)

        clip = (self.get_clip_embedding(img) * 0.3 +
                self.get_clip_embedding(dial) * 0.7)

        cnn = (self.get_cnn_embedding(img) * 0.3 +
               self.get_cnn_embedding(dial) * 0.7)

        emb = np.concatenate([clip * 0.6, cnn * 0.4])
        return emb / np.linalg.norm(emb)

    # ---------------- BUILD ----------------
    def build(self, products):
        embeddings = []
        metadata = []

        print("Building index...")

        for p in products:
            img = self.download_image(p["image"])
            if img is None:
                continue

            variants = self.generate_variants(img)

            for v in variants:
                emb = self.get_final_embedding(v)

                embeddings.append(emb)

                metadata.append({
                    **p,
                    "embedding": emb.tolist(),
                    "hash": str(imagehash.phash(v))
                })

        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        self.metadata = metadata

        faiss.write_index(self.index, "index_test3.faiss")
        json.dump(metadata, open("meta_test3.json", "w"))

        print("BUILD DONE")

    # ---------------- LOAD ----------------
    def load(self):
        self.index = faiss.read_index("index_test3.faiss")
        self.metadata = json.load(open("meta_test3.json"))

    # ---------------- SEARCH ----------------
    def search(self, image_url, top_k=5):

        img = self.download_image(image_url)
        if img is None:
            return {"success": False}

        img = self.normalize_image(img)
        dial = self.get_dial_crop(img)

        # HASH MATCH
        query_hash = imagehash.phash(img)
        for m in self.metadata:
            ref_hash = imagehash.hex_to_hash(m["hash"])
            if query_hash - ref_hash <= 5:
                return {
                    "success": True,
                    "best_match": m,
                    "exact_match": True
                }

        # EMBEDDING
        query_emb = self.get_final_embedding(img)

        D, I = self.index.search(query_emb.reshape(1, -1), 30)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx >= len(self.metadata):
                continue
            results.append(self.metadata[idx] | {"score": float(score)})

        if not results:
            return {"success": False}

        # GROUPING
        product_scores = {}
        for r in results[:10]:
            product_scores.setdefault(r["product_id"], []).append(r["score"])

        best_pid = max(product_scores, key=lambda k: sum(product_scores[k]))

        # RE-RANK
        query_color = self.get_color(dial)

        refined = []
        for r in results:
            if r["product_id"] != best_pid:
                continue

            ref_img = self.download_image(r["image"])
            if ref_img is None:
                continue

            ref_color = self.get_color(self.get_dial_crop(ref_img))
            color_score = self.color_sim(query_color, ref_color)

            final_score = r["score"] * 0.8 + color_score * 0.2

            refined.append(r | {"final_score": final_score})

        refined = sorted(refined, key=lambda x: x["final_score"], reverse=True)

        return {
            "success": True,
            "best_match": refined[0],
            "similar": refined
        }


# ---------------- INIT ----------------
service = ImageSimilarityService()
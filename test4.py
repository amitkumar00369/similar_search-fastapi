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

        import clip
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

        self.cnn_model = WatchCNN().to(self.device)

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
    def normalize(self, img):
        img = img.resize((512, 512))
        img = np.array(img)
        img = np.clip(img * 1.1, 0, 255).astype("uint8")
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return Image.fromarray(img)

    # ---------------- DIAL ----------------
    def get_dial(self, img):
        w, h = img.size
        cx, cy = w//2, h//2
        r = min(w, h)//3
        return img.crop((cx-r, cy-r, cx+r, cy+r))

    # ---------------- EMBEDDINGS ----------------
    def get_clip(self, img):
        img = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

    def get_cnn(self, img):
        img = img.resize((224, 224))
        img = torch.tensor(np.array(img)).permute(2,0,1).float()/255.0
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.cnn_model(img)
        return emb.cpu().numpy()[0]

    def get_embedding(self, img):
        img = self.normalize(img)
        dial = self.get_dial(img)

        clip = (self.get_clip(img)*0.3 + self.get_clip(dial)*0.7)
        cnn = (self.get_cnn(img)*0.3 + self.get_cnn(dial)*0.7)

        emb = np.concatenate([clip*0.6, cnn*0.4])
        return emb / np.linalg.norm(emb)

    # ---------------- BUILD ----------------
    def build(self, products):
        embeddings = []
        metadata = []

        for p in products:
            img = self.download_image(p["image"])
            if img is None:
                continue

            emb = self.get_embedding(img)

            embeddings.append(emb)
            metadata.append({
                **p,
                "embedding": emb.tolist(),
                "hash": str(imagehash.phash(img))
            })

        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        self.metadata = metadata

        faiss.write_index(self.index, "index4.faiss")
        json.dump(metadata, open("index4.json","w"))
    def load(self):
        # self.index = faiss.read_index("rolex_data.faiss")
        # self.metadata = json.load(open("rolex_data_meta2.json"))
        self.index = faiss.read_index("index4.faiss")
        self.metadata = json.load(open("index4.json"))

    # ---------------- HARD NEGATIVE ----------------
    def create_triplets(self):
        triplets = []

        for a in self.metadata:
            pos = [m for m in self.metadata if m["product_id"] == a["product_id"] and m != a]
            neg = [m for m in self.metadata if m["product_id"] != a["product_id"]]

            if pos and neg:
                triplets.append((a, pos[0], neg[0]))

        return triplets

    def triplet_loss(self, a, p, n, margin=0.3):
        pos = F.pairwise_distance(a, p)
        neg = F.pairwise_distance(a, n)
        return torch.relu(pos - neg + margin).mean()

    def train(self, epochs=3):
        optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=1e-4)

        triplets = self.create_triplets()

        for epoch in range(epochs):
            total_loss = 0

            for a, p, n in triplets:
                img_a = self.download_image(a["image"])
                img_p = self.download_image(p["image"])
                img_n = self.download_image(n["image"])

                if not img_a or not img_p or not img_n:
                    continue

                emb_a = torch.tensor(self.get_cnn(img_a)).unsqueeze(0)
                emb_p = torch.tensor(self.get_cnn(img_p)).unsqueeze(0)
                emb_n = torch.tensor(self.get_cnn(img_n)).unsqueeze(0)

                loss = self.triplet_loss(emb_a, emb_p, emb_n)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch}: Loss {total_loss}")

    # ---------------- SEARCH ----------------
    def search(self, url, top_k=5):

        img = self.download_image(url)
        if img is None:
            return {"success": False}

        img = self.normalize(img)

        # HASH FAST MATCH
        q_hash = imagehash.phash(img)
        for m in self.metadata:
            if q_hash - imagehash.hex_to_hash(m["hash"]) <= 5:
                return {"success": True, "best_match": m}

        query = self.get_embedding(img)

        D, I = self.index.search(query.reshape(1,-1), 20)

        results = []
        for score, idx in zip(D[0], I[0]):
            if score<0.50:
                continue
            if idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({
                "id": meta["id"],
                "product_id": meta["product_id"],
                "image": meta["image"],
                "score": float(score),
                "image_id": meta["image_id"],
                # "colour": meta["colour"],
                # "material": meta["material"],
                # "diameter": meta["diameter"],
                # "price": meta["price"],
                "model": meta["model"] or ""
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        if not results:
            return {
                "success": True,
                "best_match": None,
                "similar": []
            }

        return {
            "success": True,
            "best_match":  results[0],
            "similar": results
        }


# ---------------- INIT ----------------
service = ImageSimilarityService()
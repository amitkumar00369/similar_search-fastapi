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
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import imagehash
import easyocr

session = requests.Session()
reader = easyocr.Reader(['en'])

# ---------------- HASH ----------------
def get_hash(img):
    return str(imagehash.phash(img))

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

        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

        self.cnn_model = WatchCNN().to(self.device)
        self.cnn_model.eval()

        self.index = None
        self.metadata = []

    # ---------------- DOWNLOAD ----------------
    def download_image(self, url):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "image/webp,image/*,*/*;q=0.8",
                "Referer": "https://www.chrono24.com/"
            }

            resp = session.get(url, headers=headers, timeout=(3,7))

            if resp.status_code != 200:
                return None

            return Image.open(BytesIO(resp.content)).convert("RGB")
        except:
            return None

    # ---------------- CLIP ----------------
    def get_clip_embedding(self, img):
        img = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(img)
        emb = emb.cpu().numpy()
        emb = emb / np.linalg.norm(emb)
        return emb[0].astype("float32")

    # ---------------- MULTI-ANGLE ----------------
    def get_multi_angle_embedding(self, img):

        variations = [
            img,
            img.resize((img.width // 2, img.height // 2)),
            img.crop((0, 0, img.width, img.height))
        ]

        embeddings = []

        for im in variations:
            try:
                emb = self.get_clip_embedding(im)
                embeddings.append(emb)
            except:
                continue

        if not embeddings:
            return self.get_clip_embedding(img)

        final = np.mean(embeddings, axis=0)
        final = final / np.linalg.norm(final)

        return final.astype("float32")

    # ---------------- CNN ----------------
    def get_cnn_embedding(self, img):
        img = img.resize((224, 224))
        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.cnn_model(img)

        return emb.cpu().numpy()[0]

    # ---------------- BRAND DETECTION ----------------
    def detect_brand(self, img):

        brands = [
            "Rolex watch",
            "Omega watch",
            "Casio watch",
            "Seiko watch",
            "Titan watch",
            "Fossil watch"
        ]

        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        text_inputs = clip.tokenize(brands).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).softmax(dim=-1)
        best_idx = similarity.argmax().item()

        return brands[best_idx].replace(" watch", "")

    # ---------------- OCR ----------------
    def extract_reference(self, img):
        try:
            results = reader.readtext(np.array(img))

            for r in results:
                text = r[1].replace(" ", "").upper()

                if len(text) >= 5 and any(c.isdigit() for c in text):
                    return text
        except:
            pass

        return None

    # ---------------- BUILD ----------------
    def build(self, products):
        embeddings = []
        final_metadata = []

        total = len(products)
        success = 0
        failed = 0

        print(f" Total images: {total}")

        def safe_download(p):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(p["image"], headers=headers, timeout=(5, 10))

                if resp.status_code != 200:
                    return None, p

                if "image" not in resp.headers.get("Content-Type", ""):
                    return None, p

                img = Image.open(BytesIO(resp.content)).convert("RGB")
                return img, p

            except Exception:
                return None, p

        with ThreadPoolExecutor(max_workers=30) as executor:
            results = list(executor.map(safe_download, products))

        print(" Generating embeddings...")

        for i, (img, p) in enumerate(results):

            if img is None:
                failed += 1
                continue

            try:
                emb = self.get_clip_embedding(img)
                embeddings.append(emb)

                hash_val = get_hash(img)

                final_metadata.append({
                    **p,
                    "hash": str(hash_val),
                    "embedding": emb.tolist()
                })

                success += 1

            except:
                failed += 1

        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        nlist = 100 if len(embeddings) > 100 else max(1, len(embeddings)//2)

        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(
            quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
        )

        self.index.train(embeddings)
        self.index.add(embeddings)
        self.index.nprobe = min(10, nlist)

        self.metadata = final_metadata

        faiss.write_index(self.index, "index1.faiss")
        json.dump(self.metadata, open("meta1.json", "w"))

        print("DONE")

    # ---------------- LOAD ----------------
    def load(self):
        self.index = faiss.read_index("index1.faiss")
        self.metadata = json.load(open("meta1.json"))

    # ---------------- SEARCH ----------------
    def search(self, image_url, product_ids=None, top_k=5):

        img = self.download_image(image_url)
        if img is None:
            return {"success": False}

        # 🔥 HASH MATCH
        query_hash = get_hash(img)
        for m in self.metadata:
            if "hash" in m and query_hash == m["hash"]:
                return {
                    "success": True,
                    "best_match": m,
                    "exact_match": True
                }

        # 🔥 NEW FEATURES
        detected_brand = self.detect_brand(img)
        reference_number = self.extract_reference(img)

        query_clip = self.get_multi_angle_embedding(img)
        query_cnn = self.get_cnn_embedding(img)

        D, I = self.index.search(query_clip.reshape(1, -1), 100)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]

            results.append({
                "meta": meta,
                "score": float(score)
            })

        # 🔥 BRAND FILTER
        filtered = [r for r in results if r["meta"].get("brand") == detected_brand]
        if filtered:
            results = filtered

        # 🔥 VOTING
        top = results[:5]
        pid_count = {}

        for r in top:
            pid = r["meta"]["product_id"]
            pid_count[pid] = pid_count.get(pid, 0) + 1

        best_pid = max(pid_count, key=pid_count.get)

        same_product = [m for m in self.metadata if m["product_id"] == best_pid]

        # 🔥 RE-RANK
        refined = []

        for m in same_product:
            ref_img = self.download_image(m["image"])
            if ref_img is None:
                continue

            ref_clip = np.array(m["embedding"], dtype="float32")
            ref_cnn = self.get_cnn_embedding(ref_img)

            clip_sim = np.dot(query_clip, ref_clip)
            cnn_sim = np.dot(query_cnn, ref_cnn)

            final_score = (clip_sim * 0.7) + (cnn_sim * 0.3)

            refined.append({
                "id": m["id"],
                "product_id": m["product_id"],
                "image": m["image"],
                "score": float(clip_sim),
                "final_score": float(final_score)
            })

        refined = sorted(refined, key=lambda x: x["final_score"], reverse=True)

        return {
            "success": True,
            "detected_brand": detected_brand,
            "reference_number": reference_number,
            "best_match": refined[0] if refined else None,
            "similar_matches": refined[1:top_k],
            "total": len(refined)
        }


# INSTANCE
service = ImageSimilarityService()
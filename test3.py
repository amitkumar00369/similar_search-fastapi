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

session = requests.Session()

# ---------------- HASH ----------------
import imagehash

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
    def get_multi_angle_embedding(self, image):
    # simulate multi-view (crop + resize variations)

        images = [
            image,
            image.crop((0, 0, image.width, image.height)),  # original
            image.resize((image.width//2, image.height//2)),
        ]

        embeddings = []

        for img in images:
            try:
                emb = self.get_clip_embedding(img)
                embeddings.append(emb)
            except:
                continue

        if not embeddings:
            return None

        final = np.mean(embeddings, axis=0)
        final = final / np.linalg.norm(final)

        return final.astype("float32")

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

    def download_image(self, url):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "image/webp,image/*,*/*;q=0.8",
                "Referer": "https://www.chrono24.com/"
            }

            resp = session.get(url, headers=headers)

            if resp.status_code != 200:
                return None

            img = Image.open(BytesIO(resp.content)).convert("RGB")
            return img
        except:
            return None

    def get_clip_embedding(self, img):
        img = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(img)
        emb = emb.cpu().numpy()
        emb = emb / np.linalg.norm(emb)
        return emb[0].astype("float32")

    def get_cnn_embedding(self, img):
        img = img.resize((224, 224))
        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.cnn_model(img)

        return emb.cpu().numpy()[0]

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
                print(" Failed:", p["image"])
                continue

            try:
                emb = self.get_clip_embedding(img)
                embeddings.append(emb)
                hash_val = get_hash(img)

                final_metadata.append({
                    **p,
                    "hash": str(hash_val),
                    "embedding": emb.tolist()   # 🔥 ADD THIS
                })

                success += 1

            except Exception:
                failed += 1
                continue

            if (i + 1) % 50 == 0:
                print(f"Processed: {i+1}/{total} | Success: {success} | Failed: {failed}")

        if len(embeddings) == 0:
            raise Exception(" No valid embeddings generated!")

        embeddings = np.array(embeddings).astype("float32")

        print(f"Valid embeddings: {len(embeddings)}")

        dim = embeddings.shape[1]
        n_embeddings = len(embeddings)

        if n_embeddings < 100:
            nlist = max(1, n_embeddings // 2)
        else:
            nlist = 100

        print(f" Using nlist: {nlist}")

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
        print(f"Total: {total}, Success: {success}, Failed: {failed}")

    # ---------------- LOAD ----------------
    def load(self):
        self.index = faiss.read_index("index1.faiss")
        self.metadata = json.load(open("meta1.json"))

    # ---------------- SEARCH ----------------
    def search(self, image_url, product_ids=None, top_k=5):

        img = self.download_image(image_url)
        if img is None:
            return {"success": False}

        # 🔥 FAST HASH MATCH
        query_hash = get_hash(img)

        best_match = None
        best_distance = 999

        for m in self.metadata:
            if "hash" not in m:
                continue

            ref_hash = imagehash.hex_to_hash(m["hash"])
            dist = query_hash - ref_hash   # 🔥 distance

            if dist < best_distance:
                best_distance = dist
                best_match = m
        if best_distance <= 5:
            return {
                "success": True,
                "best_match": best_match,
                "similar_matches": [],
                "exact_match": True
            }

        query_clip = self.get_clip_embedding(img)
        query_cnn = self.get_cnn_embedding(img)

        D, I = self.index.search(query_clip.reshape(1, -1), 100)

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

        # 🔥 HYBRID VOTING
        top_candidates = results[:5]

        product_counts = {}
        product_scores = {}

        for r in top_candidates:
            pid = r["product_id"]
            product_counts[pid] = product_counts.get(pid, 0) + 1
            product_scores[pid] = product_scores.get(pid, 0) + r["score"]

        best_pid = None
        found_majority = any(count >= 2 for count in product_counts.values())

        if found_majority:
            best_value = -1
            for pid in product_counts:
                value = (product_counts[pid] * 1.0) + (product_scores[pid] * 0.5)
                if value > best_value:
                    best_value = value
                    best_pid = pid
        else:
            best = max(results, key=lambda x: x["score"])
            best_pid = best["product_id"]

        # FILTER SAME PRODUCT
        same_product = [
            m for m in self.metadata
            if m["product_id"] == best_pid
        ]

        # RE-RANK
        refined_results = []

        for m in same_product:
            ref_img = self.download_image(m["image"])
            if ref_img is None:
                continue

            ref_clip = np.array(m["embedding"], dtype="float32")
            ref_cnn = self.get_cnn_embedding(ref_img)

            clip_sim = np.dot(query_clip, ref_clip)
            cnn_sim = np.dot(query_cnn, ref_cnn)
            # final_score = float((clip_sim * 0.7) + (cnn_sim * 0.3))

            final_score = float((clip_sim * 0.7) + (cnn_sim * 0.3))
            # if clip_sim > 0.88:
            #     final_score += 0.1

            # #  EXTRA BOOST if almost exact
            # if clip_sim > 0.92:
            #     final_score += 0.2

            refined_results.append({
                "id": m["id"],
                "product_id": m["product_id"],
                "image": m["image"],
                "score": float(clip_sim),
                "final_score": final_score
            })

        if not refined_results:
            return {"success": False}

        # REMOVE DUPLICATES
        seen = set()
        unique_results = []

        for r in refined_results:
            if r["image"] not in seen:
                unique_results.append(r)
                seen.add(r["image"])

        refined_results = sorted(unique_results, key=lambda x: x["final_score"], reverse=True)

        return {
            "success": True,
            "best_match": refined_results[0],
            "similar_matches": refined_results[1:top_k],
            "total": len(refined_results),
            "product_id_locked": best_pid
        }


# INSTANCE
service = ImageSimilarityService()
import requests
import torch
import numpy as np
import faiss
import open_clip
from PIL import Image
from io import BytesIO
import json
import gc

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

        # 🔥 OpenCLIP BEST MODEL
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-H-14',
            pretrained='laion2b_s32b_b79k'
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        if self.device == "cpu":
            self.model = self.model.float()

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

    # ---------------- IMAGE UTILS ----------------
    def center_crop(self, img):
        w, h = img.size
        return img.crop((w//4, h//4, w*3//4, h*3//4))

    def get_dial_crop(self, img):
        w, h = img.size
        cx, cy = w // 2, h // 2
        r = min(w, h) // 3
        return img.crop((cx - r, cy - r, cx + r, cy + r))

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
        center = self.center_crop(img)

        clip = (self.get_clip_embedding(img) + self.get_clip_embedding(center)) / 2
        cnn = (self.get_cnn_embedding(img) + self.get_cnn_embedding(center)) / 2

        emb = np.concatenate([clip * 0.7, cnn * 0.3])
        return emb / np.linalg.norm(emb)

    # ---------------- BUILD ----------------
    def build(self, products):
        embeddings = []
        metadata = []

        print("Building index...")

        for i, p in enumerate(products):
            img = self.download_image(p["image"])
            if img is None:
                continue

            emb = self.get_final_embedding(img)

            embeddings.append(emb)
            metadata.append(p)

            if (i+1) % 10 == 0:
                print(f"{i+1}/{len(products)}")

        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        self.metadata = metadata

        faiss.write_index(self.index, "rolex_data2.faiss")
        json.dump(metadata, open("rolex_data_meta2.json", "w"))

        print("BUILD DONE")

    def load(self):
        self.index = faiss.read_index("rolex_data2.faiss")
        self.metadata = json.load(open("rolex_data_meta2.json"))

    # ---------------- SEARCH ----------------
    def search(self, image_url, product_ids=None, top_k=5):

        img = self.download_image(image_url)
        if img is None:
            return {"success": False}
        query_clip = self.get_clip_embedding(img)
        query_cnn = self.get_cnn_embedding(img)

        # STEP 1: GLOBAL SEARCH
        D, I = self.index.search(query_clip.reshape(1, -1), 100)

        results = []

        for score, idx in zip(D[0], I[0]):
            if score<0.77:
                continue
            if idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({
                "id": meta["id"],
                "product_id": meta["product_id"],
                "image": meta["image"],
                "score": float(score),
                "image_id": meta["image_id"]
                # "colour": meta["colour"],
                # "material": meta["material"],
                # "diameter": meta["diameter"],
                # "price": meta["price"],
                # "model": meta["model"] or ""
            })

        if not results:
            return {"success": False,
                    "message": "Not listing"}
        if len(results)==1:
            return  {
            "success": True,
            "best_match": results[0]
      
        }

        # STEP 2: TOP-K PRODUCT VOTING (FIX)
        top_candidates = results[:5]
        print("tops", top_candidates)

        product_counts = {}
        product_scores = {}

        for r in top_candidates:
            pid = r["product_id"]

            product_counts[pid] = product_counts.get(pid, 0) + 1
            product_scores[pid] = product_scores.get(pid, 0) + r["score"]

        # CHECK: any product appears ≥ 2 times
        # best_pid = None
        # found_majority = False

        # for pid, count in product_counts.items():
        #     if count >= 2:
        #         found_majority = True
        #         break

        # if found_majority:
        #     #  use voting
        #     best_value = -1

        #     for pid in product_counts:
        #         value = (product_counts[pid] * 1.0) + (product_scores[pid] * 0.5)

        #         if value > best_value:
        #             best_value = value
        #             best_pid = pid

        #     print(" Using VOTING →", best_pid)

        # else:
        #     # fallback → highest score
        #     best = max(results, key=lambda x: x["score"])
        #     best_pid = best["product_id"]
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        top1 = sorted_results[0]
        top2 = sorted_results[1] if len(sorted_results) > 1 else None
        best_pid= None

        if top2:
            diff = abs(top1["score"] - top2["score"])
            print("Score diff:", diff)
            if top1["product_id"]==top2["product_id"]:
                best_pid = top1["product_id"]

            # CONFUSION CASE
            elif diff < 0.015:
                print(" Close scores → choosing second best")
                best_pid = top2["product_id"]
            else:
                best_pid = top1["product_id"]
      

        print(" Selected PID (score):", best_pid)

        # best = max(results, key=lambda x: x["score"])
        # best_pid = best["product_id"]

        print(" Using TOP SCORE →", best_pid)

        # STEP 3: FILTER ONLY SAME PRODUCT
        same_product = [
            m for m in self.metadata
            if m["product_id"] == best_pid
        ]

        # STEP 4: RE-RANK ONLY SAME PRODUCT (CNN)
        refined_results = []

        for m in same_product:
            ref_img = self.download_image(m["image"])
            if ref_img is None:
                continue

            ref_clip = self.get_clip_embedding(ref_img)
            ref_cnn = self.get_cnn_embedding(ref_img)

            clip_sim = np.dot(query_clip, ref_clip)
            cnn_sim = np.dot(query_cnn, ref_cnn)

            final_score = float((clip_sim * 0.7) + (cnn_sim * 0.3))

            refined_results.append({
                "id": m["id"],
                "product_id": m["product_id"],
                "image": m["image"],
                "score": float(clip_sim),
                "final_score": final_score
            })

        if not refined_results:
            return {"success": False}

        # STEP 5: SORT FINAL
        refined_results = sorted(refined_results, key=lambda x: x["final_score"], reverse=True)
        best_match = next(
            (r for r in results if r["product_id"] == best_pid),
            None
        )

        return {
            "success": True,
            "best_match": best_match
            
          
        }



# INIT
service = ImageSimilarityService()
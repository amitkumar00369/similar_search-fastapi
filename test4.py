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
    # def build(self, products):
    #     embeddings = []
    #     metadata = []

    #     for p in products:
    #         img = self.download_image(p["image"])
    #         if img is None:
    #             continue

    #         emb = self.get_embedding(img)

    #         embeddings.append(emb)
    #         metadata.append({
    #             **p,
    #             "embedding": emb.tolist(),
    #             "hash": str(imagehash.phash(img))
    #         })

    #     embeddings = np.array(embeddings).astype("float32")

    #     self.index = faiss.IndexFlatIP(embeddings.shape[1])
    #     self.index.add(embeddings)

    #     self.metadata = metadata

    #     faiss.write_index(self.index, "index4.faiss")
    #     json.dump(metadata, open("index4.json","w"))
    def build(self, products, batch_size=16):
        all_embeddings = []
        all_metadata = []

        total = len(products)

        for i in range(0, total, batch_size):
            batch = products[i:i+batch_size]

            images = []
            batch_meta = []

            # ---- DOWNLOAD ----
            for p in batch:
                img = self.download_image(p["image"])
                if img is None:
                    continue

                img = self.normalize(img)

                images.append(img)
                batch_meta.append((p, img))

            if not images:
                continue

            # ---- EMBEDDINGS ----
            batch_embeddings = []
            for (p, img) in batch_meta:
                emb = self.get_embedding(img)

                batch_embeddings.append(emb)

                all_metadata.append({
                    **p,
                    "embedding": emb.tolist(),
                    "hash": str(imagehash.phash(img))
                })

            all_embeddings.extend(batch_embeddings)

            print(f"Processed batch {i//batch_size + 1} / {total//batch_size + 1}")

        # ---- FAISS ----
        embeddings = np.array(all_embeddings).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        self.metadata = all_metadata

        faiss.write_index(self.index, "index4.faiss")
        json.dump(all_metadata, open("index4.json", "w"))

        print("✅ Index built successfully!")
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
                return {"success": True, "best_match": {
                "id": m["id"],
                "product_id": m["product_id"],
                "image": m["image"],
                "score": 100,
                "image_id": m["image_id"],
                # "model": m["model"] or ""
                }
                        }

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
                # "model": meta["model"] or ""
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        top_candidates = results[:5]
        print("tops", top_candidates)

        # product_counts = {}
        # product_scores = {}

        # for r in top_candidates:
        #     pid = r["product_id"]

        #     product_counts[pid] = product_counts.get(pid, 0) + 1
        #     product_scores[pid] = product_scores.get(pid, 0) + r["score"]

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
            # if top1["product_id"]==top2["product_id"]:
            #     best_pid = top1["product_id"]

            # CONFUSION CASE
            if diff < 0.015:
                print(" Close scores → choosing second best")
                best_pid = top2["product_id"]
            else:
                best_pid = top1["product_id"]
      

        print(" Selected PID (score):", best_pid)

        # best = max(results, key=lambda x: x["score"])
        # best_pid = best["product_id"]

        print(" Using TOP SCORE →", best_pid)

        # STEP 3: FILTER ONLY SAME PRODUCT
        # same_product = [
        #     m for m in self.metadata
        #     if m["product_id"] == best_pid
        # ]

        # STEP 4: RE-RANK ONLY SAME PRODUCT (CNN)
        # refined_results = []

        # for m in same_product:
        #     ref_img = self.download_image(m["image"])
        #     if ref_img is None:
        #         continue

        #     ref_clip = self.get_clip_embedding(ref_img)
        #     ref_cnn = self.get_cnn_embedding(ref_img)

        #     clip_sim = np.dot(query_clip, ref_clip)
        #     cnn_sim = np.dot(query_cnn, ref_cnn)

        #     final_score = float((clip_sim * 0.7) + (cnn_sim * 0.3))

        #     refined_results.append({
        #         "id": m["id"],
        #         "product_id": m["product_id"],
        #         "image": m["image"],
        #         "score": float(clip_sim),
        #         "final_score": final_score
        #     })

        # if not refined_results:
        #     return {"success": False}

        # # STEP 5: SORT FINAL
        # refined_results = sorted(refined_results, key=lambda x: x["final_score"], reverse=True)
        best_match = next(
            (r for r in results if r["product_id"] == best_pid),
            None
        )
        # best_match = results[best_pid]

        return {
            "success": True,
            "best_match": best_match
            
          
        }
        if not results:
            return {
                "success": True,
                "best_match": None,
                "similar": []
            }
        print(results[:5])

        return {
            "success": True,
            "best_match":  results[0],
            "similar": results
        }


# ---------------- INIT ----------------
service = ImageSimilarityService()
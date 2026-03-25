import requests
import torch
import numpy as np
import faiss
import clip
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

class ImageSimilarityService:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # CLIP model
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.index = None
        self.metadata = []

    # ----------------------------
    # Download Image (SAFE)
    # ----------------------------
    def download_image(self, url):
        try:
            resp = requests.get(url, timeout=10)
            img = Image.open(BytesIO(resp.content))

            # fix transparency + palette issues
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            return img
        except:
            return None

    # ----------------------------
    # Batch Embeddings (SAFE)
    # ----------------------------
    def get_batch_embeddings(self, images):

        images = [img for img in images if img is not None]

        if len(images) == 0:
            return np.array([])

        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)

        with torch.no_grad():
            embeddings = self.model.encode_image(batch)

        embeddings = embeddings.cpu().numpy()

        # normalize (important for cosine similarity)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings.astype("float32")

    # ----------------------------
    # Build embeddings (FAST + ROBUST)
    # ----------------------------
    def build_embeddings(self, products, batch_size=32):

        urls = [p["image"] for p in products]

        print("⬇️ Downloading images...")

        # parallel download
        with ThreadPoolExecutor(max_workers=15) as executor:
            images = list(executor.map(self.download_image, urls))

        print("🧠 Generating embeddings...")

        all_embeddings = []
        valid_metadata = []

        for i in range(0, len(images), batch_size):

            batch_imgs = images[i:i+batch_size]
            batch_products = products[i:i+batch_size]

            batch_emb = self.get_batch_embeddings(batch_imgs)

            # skip empty batch
            if len(batch_emb) == 0:
                continue

            # filter valid images
            valid_imgs = [img for img in batch_imgs if img is not None]

            for j in range(len(valid_imgs)):
                all_embeddings.append(batch_emb[j])
                valid_metadata.append(batch_products[j])

            print(f"Processed {min(i+batch_size, len(images))}/{len(images)}")

        embeddings_array = np.array(all_embeddings).astype("float32")

        print("⚡ Building FAISS index...")

        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings_array)

        self.metadata = valid_metadata

        print("✅ Embeddings built + indexed")

    # ----------------------------
    # Save index (IMPORTANT)
    # ----------------------------
    def save(self):
        faiss.write_index(self.index, "index.faiss")

        import json
        with open("meta.json", "w") as f:
            json.dump(self.metadata, f)

        print("💾 Index saved")

    # ----------------------------
    # Load index
    # ----------------------------
    def load(self):
        import json

        self.index = faiss.read_index("index.faiss")

        with open("meta.json") as f:
            self.metadata = json.load(f)

        print("📂 Index loaded")

    # ----------------------------
    # Search (FAST)
    # ----------------------------
    def search(self, image_url, product_ids=None, top_k=5):

        if self.index is None:
            raise Exception("Index not built or loaded!")

        img = self.download_image(image_url)

        if img is None:
            return {
                "success": False,
                "error": "Invalid image URL"
            }

        emb = self.get_batch_embeddings([img])

        if len(emb) == 0:
            return {
                "success": False,
                "error": "Embedding failed"
            }

        query = emb[0].astype("float32")

        results = []

        # 🔥 STEP 1: Filter first
        if product_ids:
            product_ids = set(product_ids)

            filtered_indices = [
                i for i, item in enumerate(self.metadata)
                if item["id"] in product_ids
            ]

            if not filtered_indices:
                return {
                    "success": True,
                    "query_image": image_url,
                    "best_match": None,
                    "similar_matches": [],
                    "total_results": 0
                }

            # similarity calc
            for idx in filtered_indices:
                item_emb = self.index.reconstruct(idx)

                score = float(np.dot(query, item_emb))

                results.append({
                    "id": self.metadata[idx]["id"],
                    "product_id": self.metadata[idx]["product_id"],
                    "image": self.metadata[idx]["image"],
                    "score": round(score, 3)
                })

        else:
            # 🔥 STEP 2: Normal FAISS search
            D, I = self.index.search(query.reshape(1, -1), top_k)

            for score, idx in zip(D[0], I[0]):
                results.append({
                    "id": self.metadata[idx]["id"],
                    "product_id": self.metadata[idx]["product_id"],
                    "image": self.metadata[idx]["image"],
                    "score": float(round(score, 3))
                })

        # 🔥 STEP 3: Sort results
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        # 🔥 STEP 4: Split best + similar
        best_match = None
        similar_matches = []

        if results:
            best = results[0]

            # match type logic
            match_type = "exact" if best["score"] >= 0.99 else "similar"

            best_match = {
                **best,
                "match_type": match_type,
                "confidence": (
                    "very_high" if best["score"] > 0.95 else
                    "high" if best["score"] > 0.9 else
                    "medium"
                )
            }

            # rest similar
            for r in results[1:top_k]:
                similar_matches.append({
                    **r,
                    "match_type": "similar",
                    "confidence": (
                        "very_high" if r["score"] > 0.95 else
                        "high" if r["score"] > 0.9 else
                        "medium"
                    )
                })

        # 🔥 FINAL RESPONSE
        return {
            "success": True,
            "query_image": image_url,
            "best_match": best_match,
            "similar_matches": similar_matches,
            "total_results": len(results),
            "applied_filter": bool(product_ids)
        }


# instance
similar = ImageSimilarityService()
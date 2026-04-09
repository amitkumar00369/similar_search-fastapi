# import requests
# import torch
# import numpy as np
# import faiss
# import clip
# from PIL import Image
# from io import BytesIO
# from concurrent.futures import ThreadPoolExecutor
# from concurrent.futures import ThreadPoolExecutor, as_completed

# class ImageSimilarityService:

#     def __init__(self):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#         # CLIP model
#         self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

#         self.index = None
#         self.metadata = []

#     # ----------------------------
#     # Download Image (SAFE)
#     # ----------------------------
#     def download_image(self, url):
#         try:
#             resp = requests.get(url, timeout=(3, 7))

#             if resp.status_code != 200:
#                 return None

#             # 🔥 check content type
#             if "image" not in resp.headers.get("Content-Type", ""):
#                 return None

#             img = Image.open(BytesIO(resp.content))

#             # 🔥 verify image (important)
#             img.verify()

#             # reopen after verify
#             img = Image.open(BytesIO(resp.content))

#             if img.mode != "RGB":
#                 img = img.convert("RGB")

#             return img

#         except Exception as e:
#             # 🔥 silent skip (avoid heavy logging)
#             print("skip",url)
#             return None

#     # ----------------------------
#     # Batch Embeddings (SAFE)
#     # ----------------------------
#     def get_batch_embeddings(self, images):

#         images = [img for img in images if img is not None]

#         if len(images) == 0:
#             return np.array([])

#         batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)

#         with torch.no_grad():
#             embeddings = self.model.encode_image(batch)

#         embeddings = embeddings.cpu().numpy()
#         # print("Raw embedding shape:", embeddings)  # Debugging line

#         # normalize (important for cosine similarity)
#         embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
#         # print("Normalized embedding shape:", embeddings)  # Debugging line

#         return embeddings.astype("float32")

#     # ----------------------------
#     # Build embeddings (FAST + ROBUST)
#     # ----------------------------



#     def build_embeddings(self, products, batch_size=64):

#         print("🧠 Building embeddings (memory safe)...")

#         all_embeddings = []
#         final_metadata = []

#         for i in range(0, len(products), batch_size):

#             batch_products = products[i:i+batch_size]

#             # 🔥 Download only current batch
#             with ThreadPoolExecutor(max_workers=20) as executor:
#                 images = list(
#                     executor.map(
#                         lambda p: self.download_image(p["image"]),
#                         batch_products
#                     )
#                 )

#             # 🔥 Filter valid images
#             valid_data = [
#                 (img, prod)
#                 for img, prod in zip(images, batch_products)
#                 if img is not None
#             ]

#             if not valid_data:
#                 continue

#             batch_imgs, batch_meta = zip(*valid_data)

#             # 🔥 Generate embeddings
#             batch_emb = self.get_batch_embeddings(list(batch_imgs))

#             for j in range(len(batch_emb)):
#                 all_embeddings.append(batch_emb[j])
#                 final_metadata.append(batch_meta[j])

#             print(f"Processed {min(i+batch_size, len(products))}/{len(products)}")

#             # 🔥 FREE MEMORY (VERY IMPORTANT)
#             del images, batch_imgs, batch_meta, batch_emb
#             import gc
#             gc.collect()

#         embeddings_array = np.array(all_embeddings).astype("float32")

#         print("⚡ Building FAISS index...")

#         dimension = embeddings_array.shape[1]
#         self.index = faiss.IndexFlatIP(dimension)
#         self.index.add(embeddings_array)

#         self.metadata = final_metadata

#         print("✅ Done without crash 🚀")

#     # ----------------------------
#     # Save index (IMPORTANT)
#     # ----------------------------
#     def save(self):
#         faiss.write_index(self.index, "index.faiss")

#         import json
#         with open("meta.json", "w") as f:
#             json.dump(self.metadata, f)

#         print("💾 Index saved")

#     # ----------------------------
#     # Load index
#     # ----------------------------
#     def load(self):
#         import json

#         self.index = faiss.read_index("index.faiss")

#         with open("meta.json") as f:
#             self.metadata = json.load(f)

#         print("📂 Index loaded")

 
#     def search(self, image_url, product_ids=None, top_k=5):

#         if self.index is None:
#             raise Exception("Index not built or loaded!")

#         img = self.download_image(image_url)

#         if img is None:
#             return  {
#                     "success": True,
#                     "message": "not found",
#                     "query_image": image_url,
#                     "best_match": None,
#                     "similar_matches": [],
#                     "total_results": 0
#                 }

#         emb = self.get_batch_embeddings([img])
#         # print("Embedding shape:", emb)  # Debugging line

#         if len(emb) == 0:
#             return {
#                 "success": False,
#                 "error": "Embedding failed"
#             }

#         query = emb[0].astype("float32")
#         # print("djjjf",query)
        

#         # 🔥 STEP 1: Always search
#         search_k = 50 if product_ids else top_k
#         D, I = self.index.search(query.reshape(1, -1), search_k)

#         results = []

#         for score, idx in zip(D[0], I[0]):
#             results.append({
#                 "id": self.metadata[idx]["id"],
#                 "product_id": self.metadata[idx]["product_id"],
#                 "image": self.metadata[idx]["image"],
#                 "score": float(round(score, 3))
#             })

#         # 🔥 STEP 2: If product_ids provided → filter
#         if product_ids:
#             product_ids = set(product_ids)

#             filtered_results = [
#                 r for r in results if r["product_id"] in product_ids
#             ]

#             # ❌ No match case
#             if not filtered_results:
#                 return {
#                     "success": True,
#                     "message": "not found",
#                     "query_image": image_url,
#                     "best_match": None,
#                     "similar_matches": [],
#                     "total_results": 0
#                 }

#             # ✅ Only matched results
#             results = filtered_results

#         # 🔥 STEP 3: Sort
#         results = sorted(results, key=lambda x: x["score"], reverse=True)

#         # 🔥 STEP 4: Build response
#         best_match = None
#         similar_matches = []

#         if results:
#             best = results[0]

#             best_match = {
#                 **best,
#                 "match_type": "exact" if best["score"] >= 0.99 else "similar",
#                 "confidence": (
#                     "very_high" if best["score"] > 0.95 else
#                     "high" if best["score"] > 0.9 else
#                     "medium"
#                 )
#             }

#             for r in results[1:top_k]:
#                 similar_matches.append({
#                     **r,
#                     "match_type": "similar",
#                     "confidence": (
#                         "very_high" if r["score"] > 0.95 else
#                         "high" if r["score"] > 0.9 else
#                         "medium"
#                     )
#                 })

#         return {
#             "success": True,
#             "query_image": image_url,
#             "best_match": best_match,
#             "similar_matches": similar_matches,
#             "total_results": len(results),
#             "filter_applied": bool(product_ids)
#         }


# # instance
# similar = ImageSimilarityService()

import requests
import torch
import numpy as np
import faiss
import clip
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import gc
import json


class ImageSimilarityService:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # CLIP model
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

        self.index = None
        self.metadata = []

    # ----------------------------
    # Download Image (SAFE)
    # ----------------------------
    def download_image(self, url):
        try:
            resp = requests.get(url, timeout=(3, 7))

            if resp.status_code != 200:
                return None

            if "image" not in resp.headers.get("Content-Type", ""):
                return None

            img = Image.open(BytesIO(resp.content))
            img.verify()

            img = Image.open(BytesIO(resp.content))

            if img.mode != "RGB":
                img = img.convert("RGB")

            return img

        except Exception:
            print("skip", url)
            return None

    # ----------------------------
    # Batch Embeddings
    # ----------------------------
    def get_batch_embeddings(self, images):

        images = [img for img in images if img is not None]

        if len(images) == 0:
            return np.array([])

        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)

        with torch.no_grad():
            embeddings = self.model.encode_image(batch)

        embeddings = embeddings.cpu().numpy()

        # normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings.astype("float32")

    # ----------------------------
    # Build embeddings
    # ----------------------------
    def build_embeddings(self, products, batch_size=64):

        print("🧠 Building embeddings (memory safe)...")

        all_embeddings = []
        final_metadata = []

        for i in range(0, len(products), batch_size):

            batch_products = products[i:i+batch_size]

            with ThreadPoolExecutor(max_workers=20) as executor:
                images = list(
                    executor.map(
                        lambda p: self.download_image(p["image"]),
                        batch_products
                    )
                )

            valid_data = [
                (img, prod)
                for img, prod in zip(images, batch_products)
                if img is not None
            ]

            if not valid_data:
                continue

            batch_imgs, batch_meta = zip(*valid_data)

            batch_emb = self.get_batch_embeddings(list(batch_imgs))

            for j in range(len(batch_emb)):
                all_embeddings.append(batch_emb[j])
                final_metadata.append(batch_meta[j])

            print(f"Processed {min(i+batch_size, len(products))}/{len(products)}")

            del images, batch_imgs, batch_meta, batch_emb
            gc.collect()

        embeddings_array = np.array(all_embeddings).astype("float32")

        print("⚡ Building FAISS index...")

        dimension = embeddings_array.shape[1]

        # 🔥 Chrono24 style IVF index
        nlist = 100
        quantizer = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

        self.index.train(embeddings_array)
        self.index.add(embeddings_array)

        self.index.nprobe = 10

        self.metadata = final_metadata

        print("✅ Done without crash 🚀")

    # ----------------------------
    # Save index
    # ----------------------------
    def save(self):
        faiss.write_index(self.index, "index.faiss")

        with open("meta.json", "w") as f:
            json.dump(self.metadata, f)

        print("💾 Index saved")

    # ----------------------------
    # Load index
    # ----------------------------
    def load(self):
        self.index = faiss.read_index("index.faiss")

        with open("meta.json") as f:
            self.metadata = json.load(f)

        print("📂 Index loaded")

    # ----------------------------
    # Search (Chrono24 logic)
    # ----------------------------
    def search(self, image_url, product_ids=None, top_k=5):

        if self.index is None:
            raise Exception("Index not built or loaded!")

        img = self.download_image(image_url)

        if img is None:
            return {
                "success": True,
                "message": "not found",
                "query_image": image_url,
                "best_match": None,
                "similar_matches": [],
                "total_results": 0
            }

        emb = self.get_batch_embeddings([img])

        if len(emb) == 0:
            return {
                "success": False,
                "error": "Embedding failed"
            }

        query = emb[0].astype("float32")

        # 🔥 Multi-stage search
        search_k = 100 if product_ids else max(50, top_k)
        D, I = self.index.search(query.reshape(1, -1), search_k)

        results = []

        for score, idx in zip(D[0], I[0]):
            if idx >= len(self.metadata):
                continue

            results.append({
                "id": self.metadata[idx]["id"],
                "product_id": self.metadata[idx]["product_id"],
                "image": self.metadata[idx]["image"],
                "score": float(round(score, 4))
            })

        # 🔥 Confidence threshold
        MIN_SCORE = 0.75
        results = [r for r in results if r["score"] >= MIN_SCORE]

        # 🔥 Filter by product_ids (existing logic)
        if product_ids:
            product_ids = set(product_ids)

            results = [
                r for r in results if r["product_id"] in product_ids
            ]

            if not results:
                return {
                    "success": True,
                    "message": "not found",
                    "query_image": image_url,
                    "best_match": None,
                    "similar_matches": [],
                    "total_results": 0
                }

        # 🔥 Remove duplicates
        seen_products = set()
        unique_results = []

        for r in results:
            if r["product_id"] not in seen_products:
                unique_results.append(r)
                seen_products.add(r["product_id"])

        results = unique_results

        # 🔥 Re-ranking
        for r in results:
            score = r["score"]

            if product_ids and r["product_id"] in product_ids:
                score += 0.05

            r["final_score"] = round(score, 4)

        results = sorted(results, key=lambda x: x["final_score"], reverse=True)

        if not results:
            return {
                "success": True,
                "message": "not found",
                "query_image": image_url,
                "best_match": None,
                "similar_matches": [],
                "total_results": 0
            }

        # 🔥 Best match
        best = results[0]

        best_match = {
            **best,
            "match_type": "exact" if best["score"] >= 0.97 else "similar",
            "confidence": (
                "very_high" if best["score"] > 0.95 else
                "high" if best["score"] > 0.9 else
                "medium"
            )
        }

        # 🔥 Similar matches
        similar_matches = []

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

        return {
            "success": True,
            "query_image": image_url,
            "best_match": best_match,
            "similar_matches": similar_matches,
            "total_results": len(results),
            "filter_applied": bool(product_ids)
        }


# instance
similar = ImageSimilarityService()
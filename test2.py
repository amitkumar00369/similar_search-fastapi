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
import aiohttp
import asyncio
import time

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
    async def async_download(self, session, p, retries=3):
        url = p["image"]

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/webp,image/*,*/*;q=0.8",
            "Referer": "https://www.chrono24.com/"
        }

        for attempt in range(retries):
            try:
                async with session.get(url, headers=headers, timeout=10) as resp:

                    if resp.status != 200:
                        reason = f"status_{resp.status}"
                        continue

                    content_type = resp.headers.get("Content-Type", "")
                    if "image" not in content_type:
                        return None, p, "not_image"

                    data = await resp.read()

                    try:
                        img = Image.open(BytesIO(data)).convert("RGB")
                        return img, p, None
                    except:
                        return None, p, "corrupt_image"

            except asyncio.TimeoutError:
                reason = "timeout"
            except Exception:
                reason = "connection_error"

            await asyncio.sleep(1 * (attempt + 1))

        return None, p, reason
    async def download_batch(self, batch_products):

        connector = aiohttp.TCPConnector(limit=5)  # control concurrency

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self.async_download(session, p)
                for p in batch_products
            ]
            return await asyncio.gather(*tasks)

    def download_image(self, url):
        try:
            # resp = requests.get(url, timeout=(3, 7))
            # print(resp.status_code)
            # print(resp.headers)
            # print(resp)
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
    # def build(self, products):
    #     embeddings = []
    #     final_metadata = []

    #     total = len(products)
    #     success = 0
    #     failed = 0

    #     print(f" Total images: {total}")

    #     # improved download with retry + headers
    #     def safe_download(p):
    #         try:
    #             headers = {"User-Agent": "Mozilla/5.0"}
    #             resp = requests.get(p["image"], headers=headers, timeout=(3, 7))

    #             if resp.status_code != 200:
    #                 return None, p

    #             if "image" not in resp.headers.get("Content-Type", ""):
    #                 return None, p

    #             img = Image.open(BytesIO(resp.content)).convert("RGB")
    #             return img, p

    #         except Exception:
    #             return None, p

    #     #  parallel download
    #     with ThreadPoolExecutor(max_workers=10) as executor:
    #         results = list(executor.map(safe_download, products))

    #     print(" Generating embeddings...")

    #     for i, (img, p) in enumerate(results):

    #         if img is None:
    #             failed += 1
    #             # debug failed images
    #             print(" Failed:", p["image"])
    #             continue

    #         try:
    #             emb = self.get_clip_embedding(img)
    #             embeddings.append(emb)
    #             final_metadata.append(p)
    #             success += 1
    #         except Exception:
    #             failed += 1
    #             continue

    #         # progress log
    #         if (i + 1) % 32 == 0:
    #             print(f"Processed: {i+1}/{total} | Success: {success} | Failed: {failed}")

    #     if len(embeddings) == 0:
    #         raise Exception(" No valid embeddings generated!")

    #     embeddings = np.array(embeddings).astype("float32")

    #     print(f"Valid embeddings: {len(embeddings)}")

    #     dim = embeddings.shape[1]

    #     # FIX: dynamic nlist (avoid FAISS crash)
    #     n_embeddings = len(embeddings)

    #     if n_embeddings < 100:
    #         nlist = max(1, n_embeddings // 2)
    #     else:
    #         nlist = 100

    #     print(f" Using nlist: {nlist}")

    #     quantizer = faiss.IndexFlatIP(dim)
    #     self.index = faiss.IndexIVFFlat(
    #         quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
    #     )

    #     self.index.train(embeddings)
    #     self.index.add(embeddings)
    #     self.index.nprobe = min(10, nlist)

    #     self.metadata = final_metadata

    #     faiss.write_index(self.index, "rolex.faiss")
    #     json.dump(self.metadata, open("rolex_meta2.json", "w"))

    #     print("DONE")
    #     print(f"Total: {total}, Success: {success}, Failed: {failed}")
    def build(self, products):

        embeddings = []
        final_metadata = []
        failedImage = []

        total = len(products)
        success = 0
        failed = 0

        print(f"Total images: {total}")

        def safe_download(p):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(p["image"], headers=headers, timeout=(3, 7))

                if resp.status_code != 200:
                    # failedImage.append(resp)
                    
                    return None, p

                if "image" not in resp.headers.get("Content-Type", ""):
                    failedImage.append(resp)
                    return None, p
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                return img, p

            except Exception:
                # print(resp)
                # failedImage.append(resp)
                return None, p

        # CONFIG (IMPORTANT)
        batch_size = 16
        max_workers = 3

        print("Downloading + Processing in batches...")

        for i in range(0, total, batch_size):

            batch_products = products[i:i + batch_size]

            # download batch
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(safe_download, batch_products))

            # process batch
            for img, p in results:

                if img is None:
                    failed += 1
                    continue

                try:
                    emb = self.get_clip_embedding(img)
                    embeddings.append(emb)
                    final_metadata.append(p)
                    success += 1
                except Exception:
                    failed += 1
                    continue

            print(f"Processed: {min(i+batch_size, total)}/{total} | Success: {success} | Failed: {failed}")

            # 🔥 free memory after each batch
            del results
            gc.collect()

        if len(embeddings) == 0:
            raise Exception("No valid embeddings generated!")

        embeddings = np.array(embeddings).astype("float32")

        print(f"Valid embeddings: {len(embeddings)}")

        dim = embeddings.shape[1]

        n_embeddings = len(embeddings)

        if n_embeddings < 100:
            nlist = max(1, n_embeddings // 2)
        else:
            nlist = 100

        print(f"Using nlist: {nlist}")

        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(
            quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
        )

        self.index.train(embeddings)
        self.index.add(embeddings)
        self.index.nprobe = min(10, nlist)

        self.metadata = final_metadata

        faiss.write_index(self.index, "rolex_data.faiss")
        json.dump(self.metadata, open("rolex_data_meta2.json", "w"))

        print("DONE")
        print(f"Total: {total}, Success: {success}, Failed: {failed}")

    # ---------------- LOAD ----------------
    def load(self):
        self.index = faiss.read_index("rolex_data.faiss")
        self.metadata = json.load(open("rolex_data_meta2.json"))

    # ---------------- SEARCH ----------------
    # def search(self, image_url, product_ids=None, top_k=5):

    #     img = self.download_image(image_url)
    #     if img is None:
    #         print("yeyyyegdgdbh hhhhfhfjhf img bbd d")
    #         return {"success": False}

    #     query_clip = self.get_clip_embedding(img)
    #     query_cnn = self.get_cnn_embedding(img)

    #     D, I = self.index.search(query_clip.reshape(1, -1), 100)

    #     results = []

    #     for score, idx in zip(D[0], I[0]):
    #         if idx >= len(self.metadata):
    #             continue

    #         meta = self.metadata[idx]
    #         # print("meta",meta)

    #         results.append({
    #             "id": meta["id"],
    #             "product_id": meta["product_id"],
    #             "image": meta["image"],
    #             "score": float(score)
    #         })

    #     # filter product_ids
    #     if product_ids:
    #         product_ids = set(product_ids)
    #         results = [r for r in results if r["product_id"] in product_ids]

    #     # deduplicate
    #     seen = set()
    #     unique = []
    #     for r in results:
    #         if r["product_id"] not in seen:
    #             unique.append(r)
    #             seen.add(r["product_id"])

    #     results = unique

    #     # CNN re-ranking
    #     for r in results[:20]:
    #         ref_img = self.download_image(r["image"])
    #         if ref_img is None:
    #             continue

    #         ref_emb = self.get_cnn_embedding(ref_img)
    #         sim = np.dot(query_cnn, ref_emb)

    #         r["final_score"] = float(r["score"] + sim * 0.2)

    #     results = sorted(results, key=lambda x: x.get("final_score", x["score"]), reverse=True)

    #     return {
    #         "success": True,
    #         "best_match": results[0] if results else None,
    #         "similar_matches": results[1:top_k],
    #         "total": len(results)
    #     }
    # def search(self, image_url, product_ids=None, top_k=5):

    #     img = self.download_image(image_url)
    #     if img is None:
    #         return {"success": False}

    #     query_clip = self.get_clip_embedding(img)
    #     query_cnn = self.get_cnn_embedding(img)

    #     D, I = self.index.search(query_clip.reshape(1, -1), 100)

    #     results = []

    #     for score, idx in zip(D[0], I[0]):
    #         if idx >= len(self.metadata):
    #             continue

    #         meta = self.metadata[idx]

    #         results.append({
    #             "id": meta["id"],
    #             "product_id": meta["product_id"],
    #             "image": meta["image"],
    #             "score": float(score)
    #         })

    #     if not results:
    #         return {"success": False}

    #     #  CNN re-ranking
    #     for r in results[:20]:
    #         ref_img = self.download_image(r["image"])
    #         if ref_img is None:
    #             continue

    #         ref_emb = self.get_cnn_embedding(ref_img)
    #         sim = np.dot(query_cnn, ref_emb)

    #         r["final_score"] = float(r["score"] + sim * 0.2)

    #     results = sorted(results, key=lambda x: x.get("final_score", x["score"]), reverse=True)

    #     # STEP 1: BEST MATCH
    #     best = results[0]
    #     best_pid = best["product_id"]

    #     #  STEP 2: FILTER SAME PRODUCT
    #     same_product_results = [r for r in results if r["product_id"] == best_pid]

    #     #  STEP 3: REMOVE DUPLICATE IDS (optional)
    #     seen = set()
    #     unique = []
    #     for r in same_product_results:
    #         if r["id"] not in seen:
    #             unique.append(r)
    #             seen.add(r["id"])

    #     #  STEP 4: FINAL OUTPUT
    #     return {
    #         "success": True,
    #         "best_match": best,
    #         "similar_matches": unique[1:top_k],  # same product only
    #         "total": len(unique)
    #     }
    # def search(self, image_url, product_ids=None, top_k=5):

    #     img = self.download_image(image_url)
    #     if img is None:
    #         return {"success": False}

    #     query_clip = self.get_clip_embedding(img)
    #     query_cnn = self.get_cnn_embedding(img)

    #     # STEP 1: GLOBAL SEARCH
    #     D, I = self.index.search(query_clip.reshape(1, -1), 100)

    #     results = []

    #     for score, idx in zip(D[0], I[0]):
    #         if idx >= len(self.metadata):
    #             continue

    #         meta = self.metadata[idx]

    #         results.append({
    #             "id": meta["id"],
    #             "product_id": meta["product_id"],
    #             "image": meta["image"],
    #             "score": float(score)
    #         })

    #     if not results:
    #         return {"success": False}

    #     # STEP 2: FIND BEST MATCH
    #     best = max(results, key=lambda x: x["score"])
    #     best_pid = best["product_id"]
    #     print("bnm,hvbnmvbn",best_pid)

    #     # STEP 3: FILTER ONLY SAME PRODUCT
    #     same_product = [
    #         m for m in self.metadata
    #         if m["product_id"] == best_pid
    #     ]

    #     # STEP 4: RE-RANK ONLY SAME PRODUCT (CNN)
    #     refined_results = []

    #     for m in same_product:
    #         ref_img = self.download_image(m["image"])
    #         if ref_img is None:
    #             continue

    #         ref_clip = self.get_clip_embedding(ref_img)
    #         ref_cnn = self.get_cnn_embedding(ref_img)

    #         #  hybrid score
    #         clip_sim = np.dot(query_clip, ref_clip)
    #         cnn_sim = np.dot(query_cnn, ref_cnn)

    #         final_score = float(clip_sim + cnn_sim * 0.3)

    #         refined_results.append({
    #             "id": m["id"],
    #             "product_id": m["product_id"],
    #             "image": m["image"],
    #             "score": float(clip_sim),
    #             "final_score": final_score
    #         })

    #     #  STEP 5: SORT FINAL
    #     refined_results = sorted(refined_results, key=lambda x: x["final_score"], reverse=True)

    #     return {
    #         "success": True,
    #         "best_match": refined_results[0] if refined_results else None,
    #         "similar_matches": refined_results[1:top_k],
    #         "total": len(refined_results),
    #         "product_id_locked": best_pid  #  important
    #     }
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
                "score": float(score)
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
            # "best_match": refined_results[0],
            "best_match": results
            
            # "similar_matches": top_candidates[1:2],
            # "total": len(refined_results),
            # "product_id_locked": best_pid
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

            # 🔥 CONFUSION CASE
            if diff < 0.01:
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
            # "best_match": refined_results[0],
            "best_match": best_match
            
            # "similar_matches": top_candidates[1:2],
            # "total": len(refined_results),
            # "product_id_locked": best_pid
        }


# ---------------- FASTAPI ----------------
# app = FastAPI()
service = ImageSimilarityService()

# class BuildRequest(BaseModel):
#     products: list

# class SearchRequest(BaseModel):
#     url: str
#     product_ids: list = []

# @app.post("/build")
# def build(req: BuildRequest):
#     service.build(req.products)
#     return {"status": "built"}

# @app.post("/load")
# def load():
#     service.load()
#     return {"status": "loaded"}

# @app.post("/search")
# def search(req: SearchRequest):
#     return service.search(req.url, req.product_ids)
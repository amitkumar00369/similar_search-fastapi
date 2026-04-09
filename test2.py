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
    def build(self, products):
        embeddings = []
        final_metadata = []

        total = len(products)
        success = 0
        failed = 0

        print(f" Total images: {total}")

        # improved download with retry + headers
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

        #  parallel download
        with ThreadPoolExecutor(max_workers=30) as executor:
            results = list(executor.map(safe_download, products))

        print(" Generating embeddings...")

        for i, (img, p) in enumerate(results):

            if img is None:
                failed += 1
                # debug failed images
                print(" Failed:", p["image"])
                continue

            try:
                emb = self.get_clip_embedding(img)
                embeddings.append(emb)
                final_metadata.append(p)
                success += 1
            except Exception:
                failed += 1
                continue

            # progress log
            if (i + 1) % 50 == 0:
                print(f"Processed: {i+1}/{total} | Success: {success} | Failed: {failed}")

        if len(embeddings) == 0:
            raise Exception(" No valid embeddings generated!")

        embeddings = np.array(embeddings).astype("float32")

        print(f"Valid embeddings: {len(embeddings)}")

        dim = embeddings.shape[1]

        # FIX: dynamic nlist (avoid FAISS crash)
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
            print("yeyyyegdgdbh hhhhfhfjhf img bbd d")
            return {"success": False}

        query_clip = self.get_clip_embedding(img)
        query_cnn = self.get_cnn_embedding(img)

        D, I = self.index.search(query_clip.reshape(1, -1), 100)

        results = []

        for score, idx in zip(D[0], I[0]):
            if idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]
            # print("meta",meta)

            results.append({
                "id": meta["id"],
                "product_id": meta["product_id"],
                "image": meta["image"],
                "score": float(score)
            })

        # filter product_ids
        if product_ids:
            product_ids = set(product_ids)
            results = [r for r in results if r["product_id"] in product_ids]

        # deduplicate
        seen = set()
        unique = []
        for r in results:
            if r["product_id"] not in seen:
                unique.append(r)
                seen.add(r["product_id"])

        results = unique

        # CNN re-ranking
        for r in results[:20]:
            ref_img = self.download_image(r["image"])
            if ref_img is None:
                continue

            ref_emb = self.get_cnn_embedding(ref_img)
            sim = np.dot(query_cnn, ref_emb)

            r["final_score"] = float(r["score"] + sim * 0.2)

        results = sorted(results, key=lambda x: x.get("final_score", x["score"]), reverse=True)

        return {
            "success": True,
            "best_match": results[0] if results else None,
            "similar_matches": results[1:top_k],
            "total": len(results)
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
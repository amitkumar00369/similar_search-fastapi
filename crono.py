import requests
import torch
import numpy as np
import faiss
import clip
from PIL import Image
from io import BytesIO
import json
import gc
import aiohttp
import asyncio
import cv2
from collections import Counter

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
from ultralytics import YOLO

class ImageSimilarityService:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

        self.cnn_model = WatchCNN().to(self.device)
        self.cnn_model.eval()

        self.yolo_model = YOLO("yolov8n.pt")

        self.index = None
        self.metadata = []

    # ---------------- PREPROCESS ----------------
    def preprocess_image(self, img):
        try:
            img = img.resize((512, 512))
            img = np.array(img)

            img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
            img = cv2.GaussianBlur(img, (3, 3), 0)

            return Image.fromarray(img)
        except:
            return img

    # ---------------- DETECT WATCH ----------------
    def detect_watch(self, img):
        try:
            results = self.yolo_model(np.array(img))

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    return img.crop((x1, y1, x2, y2))

            return img
        except:
            return img

    # ---------------- ASYNC DOWNLOAD ----------------
    async def async_download(self, session, p, retries=3):
        url = p["image"]

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/webp,image/*,*/*;q=0.8",
            "Referer": "https://www.chrono24.com/"
        }

        last_reason = None

        for attempt in range(retries):
            try:
                async with session.get(url, headers=headers, timeout=10) as resp:

                    if resp.status != 200:
                        last_reason = f"status_{resp.status}"
                        continue

                    if "image" not in resp.headers.get("Content-Type", ""):
                        return None, p, "not_image"

                    data = await resp.read()

                    try:
                        img = Image.open(BytesIO(data)).convert("RGB")
                        return img, p, None
                    except:
                        return None, p, "corrupt_image"

            except asyncio.TimeoutError:
                last_reason = "timeout"
            except Exception:
                last_reason = "connection_error"

            await asyncio.sleep(1 * (attempt + 1))

        return None, p, last_reason

    async def download_batch(self, batch_products):
        connector = aiohttp.TCPConnector(limit=5)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.async_download(session, p) for p in batch_products]
            return await asyncio.gather(*tasks)

    # ---------------- FALLBACK DOWNLOAD ----------------
    def download_image(self, url):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Accept": "image/webp,image/*,*/*;q=0.8",
                "Referer": "https://www.chrono24.com/"
            }

            resp = session.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return None

            return Image.open(BytesIO(resp.content)).convert("RGB")
        except:
            return None

    # ---------------- EMBEDDINGS ----------------
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

        print(f"Total images: {total}")

        batch_size = 8

        for i in range(0, total, batch_size):

            batch_products = products[i:i + batch_size]
            results = asyncio.run(self.download_batch(batch_products))

            for img, p, reason in results:

                if img is None:
                    failed += 1
                    continue

                try:
                    img = self.preprocess_image(img)
                    img = self.detect_watch(img)
                    dial = self.detect_dial(img)

                    clip_watch = self.get_clip_embedding(img)
                    clip_dial = self.get_clip_embedding(dial)

                    cnn_watch = self.get_cnn_embedding(img)
                    cnn_dial = self.get_cnn_embedding(dial)

                    emb = np.concatenate([
                        clip_watch * 0.5,
                        clip_dial * 0.2,
                        cnn_watch * 0.2,
                        cnn_dial * 0.1
                    ])

                    embeddings.append(emb)
                    final_metadata.append(p)
                    success += 1

                except:
                    failed += 1
                    continue

            print(f"Processed: {min(i+batch_size, total)}/{total} | Success: {success} | Failed: {failed}")

            gc.collect()

        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]

        nlist = max(1, min(100, len(embeddings)//2))

        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        self.index.train(embeddings)
        self.index.add(embeddings)
        self.index.nprobe = min(10, nlist)

        self.metadata = final_metadata

        faiss.write_index(self.index, "crono.faiss")
        json.dump(self.metadata, open("cron_data.json", "w"))

        print("DONE")
    def load(self):
        self.index = faiss.read_index("crono.faiss")
        self.metadata = json.load(open("cron_data.json"))

    # ---------------- SEARCH ----------------
    def detect_dial(self, img):
        h, w = img.size[1], img.size[0]

        # center crop (simple but powerful)
        cx, cy = w//2, h//2
        size = int(min(w, h) * 0.6)

        return img.crop((cx-size//2, cy-size//2, cx+size//2, cy+size//2))
    def search(self, image_url, product_ids=None, top_k=5):

        img = self.download_image(image_url)
        if img is None:
            return {"success": False}

        img = self.preprocess_image(img)
        img = self.detect_watch(img)
        img = self.detect_dial(img)
        

        query_clip = self.get_clip_embedding(img)
        query_cnn = self.get_cnn_embedding(img)

        query_emb = np.concatenate([query_clip * 0.7, query_cnn * 0.3])

        D, I = self.index.search(query_emb.reshape(1, -1), 50)

        results = []

        for score, idx in zip(D[0], I[0]):
            if idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]

            results.append({
                "id": meta["id"],
                "product_id": meta["product_id"],
                "image": meta["image"],
                "score": float(score),
                "confidence": round(float(score) * 100, 2)
            })

        if not results:
            return {"success": False}
        print(results[:5])

        # 🔥 PRODUCT VOTING (Chrono24 logic)
        top_pids = [r["product_id"] for r in results[:10]]
        best_pid = Counter(top_pids).most_common(1)[0][0]

        # filter same product
        same_product = [m for m in self.metadata if m["product_id"] == best_pid]

        refined = []

        for m in same_product:
            ref_img = self.download_image(m["image"])
            if ref_img is None:
                continue

            ref_img = self.preprocess_image(ref_img)
            ref_img = self.detect_watch(ref_img)

            ref_clip = self.get_clip_embedding(ref_img)
            ref_cnn = self.get_cnn_embedding(ref_img)

            clip_sim = np.dot(query_clip, ref_clip)
            cnn_sim = np.dot(query_cnn, ref_cnn)

            final_score = float((clip_sim * 0.7) + (cnn_sim * 0.3))

            refined.append({
                "id": m["id"],
                "product_id": m["product_id"],
                "image": m["image"],
                "final_score": final_score
            })

        refined = sorted(refined, key=lambda x: x["final_score"], reverse=True)

        return {
            "success": True,
            "best_match": refined[0] if refined else None,
            "similar_matches": refined[1:top_k],
            "total": len(refined)
        }
        
service = ImageSimilarityService()
import requests
import torch
import numpy as np
import faiss
import clip
from PIL import Image
from io import BytesIO
import json
import cv2
from show_img import showImage

from ultralytics import YOLO

# ---------------- CNN ----------------
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

        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.cnn_model = WatchCNN().to(self.device).eval()

        self.yolo_model = YOLO("yolov8m.pt")

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

    # ---------------- YOLO ----------------
    def detect_watch(self, img):
        try:
            results = self.yolo_model(np.array(img), verbose=False)

            best_box = None
            best_area = 0

            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) != 74:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)

                    if area > best_area:
                        best_area = area
                        best_box = (x1, y1, x2, y2)

            if best_box is None:
                return img

            x1, y1, x2, y2 = best_box

            w, h = img.size
            dx = int((x2 - x1) * 0.05)
            dy = int((y2 - y1) * 0.05)

            return img.crop((
                max(0, x1-dx),
                max(0, y1-dy),
                min(w, x2+dx),
                min(h, y2+dy)
            ))

        except:
            return img

    # ---------------- UTILS ----------------
    def center_crop(self, img):
        w, h = img.size
        return img.crop((w//4, h//4, w*3//4, h*3//4))

    def is_zoomed(self, original, crop):
        ow, oh = original.size
        cw, ch = crop.size
        return (cw * ch) / (ow * oh) > 0.7

    # ---------------- IMAGE PROCESS ----------------
    def enhance_image(self, img):
        img_np = np.array(img)

        img_np = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)

        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        img_np = cv2.filter2D(img_np, -1, kernel)

        return Image.fromarray(img_np)

    def add_blur(self, img):
        img_np = np.array(img)
        img_np = cv2.GaussianBlur(img_np, (5,5), 0)
        return Image.fromarray(img_np)

    # ---------------- EMBEDDING ----------------
    def get_clip_embedding(self, img):
        img = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_image(img)
        emb = emb.cpu().numpy()[0]
        return emb / np.linalg.norm(emb)

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
        emb = emb / np.linalg.norm(emb)

        return emb.astype("float32")

    # ---------------- BUILD ----------------
    def build(self, products):

        embeddings = []
        metadata = []

        print("Building index...")

        for i, p in enumerate(products):

            img = self.download_image(p["image"])
            # showImage(img)
            if img is None:
                continue

            watch = self.detect_watch(img)
            # showImage(watch)

            # ✅ original
            emb1 = self.get_final_embedding(watch)

            # ✅ blurred augmentation
            blur = self.add_blur(watch)
            # showImage(blur)
            emb2 = self.get_final_embedding(blur)

            emb = (emb1 + emb2) / 2
            emb = emb / np.linalg.norm(emb)

            embeddings.append(emb)
            metadata.append(p)

            if (i+1) % 10 == 0:
                print(f"{i+1}/{len(products)}")

        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        self.metadata = metadata

        faiss.write_index(self.index, "index_final.faiss")
        json.dump(metadata, open("meta_final.json", "w"))

        print("BUILD DONE")
    def load(self):
        self.index = faiss.read_index("index_final.faiss")
        self.metadata = json.load(open("meta_final.json"))

    # ---------------- SEARCH ----------------
    # def search(self, image_url, top_k=5):

    #     img = self.download_image(image_url)
    #     if img is None:
    #         return {"success": False}

    #     # ✅ only here enhance
    #     img = self.enhance_image(img)

    #     watch = self.detect_watch(img)

    #     if self.is_zoomed(img, watch):
    #         final = watch
    #     else:
    #         final = self.center_crop(watch)

    #     emb1 = self.get_final_embedding(img)
    #     emb2 = self.get_final_embedding(watch)
    #     emb3 = self.get_final_embedding(final)

    #     query = (emb1 + emb2 + emb3) / 3
    #     query = query / np.linalg.norm(query)

    #     D, I = self.index.search(query.reshape(1, -1), 20)

    #     results = []

    #     for score, idx in zip(D[0], I[0]):
    #         if idx >= len(self.metadata):
    #             continue

    #         results.append({
    #             "id": self.metadata[idx]["id"],
    #             "product_id": self.metadata[idx]["product_id"],
    #             "image": self.metadata[idx]["image"],
    #             "score": float(score)
    #         })

    #     results = sorted(results, key=lambda x: x["score"], reverse=True)

    #     return {
    #         "success": True,
    #         "best_match": results[0] if results else None,
    #         "similar": results
    #     }
    def search(self, image_url, top_k=5):

        img = self.download_image(image_url)
        if img is None:
            return {"success": False}

        # ✅ enhance only for query
        img = self.enhance_image(img)

        watch = self.detect_watch(img)

        if self.is_zoomed(img, watch):
            final = watch
        else:
            final = self.center_crop(watch)

        # 🔥 multi-view embedding
        emb1 = self.get_final_embedding(img)
        emb2 = self.get_final_embedding(watch)
        emb3 = self.get_final_embedding(final)

        query = (emb1 + emb2 + emb3) / 3
        query = query / np.linalg.norm(query)

        # 🔍 FAISS search
        D, I = self.index.search(query.reshape(1, -1), 30)

        results = []

        for score, idx in zip(D[0], I[0]):
            if idx >= len(self.metadata):
                continue

            results.append({
                "id": self.metadata[idx]["id"],
                "product_id": self.metadata[idx]["product_id"],
                "image": self.metadata[idx]["image"],
                "score": float(score)
            })

        if not results:
            return {"success": False}

        # ---------------- 🔥 STEP 1: GROUP BY PRODUCT ----------------
        product_scores = {}

        for r in results:
            pid = r["product_id"]
            product_scores.setdefault(pid, []).append(r["score"])

        # average score per product
        product_avg = {
            pid: sum(scores) / len(scores)
            for pid, scores in product_scores.items()
        }

        # best product id
        best_pid = max(product_avg, key=product_avg.get)

        # ---------------- 🔥 STEP 2: FILTER SAME PRODUCT ----------------
        same_product = [
            r for r in results if r["product_id"] == best_pid
        ]

        # ---------------- 🔥 STEP 3: CNN RE-RANK ----------------
        refined = []

        query_cnn = self.get_cnn_embedding(final)

        for r in same_product:
            ref_img = self.download_image(r["image"])
            if ref_img is None:
                continue

            ref_cnn = self.get_cnn_embedding(ref_img)

            cnn_sim = np.dot(query_cnn, ref_cnn)

            final_score = (r["score"] * 0.7) + (cnn_sim * 0.3)

            refined.append({
                **r,
                "final_score": float(final_score)
            })

        if not refined:
            return {"success": False}

        # ---------------- 🔥 STEP 4: SORT FINAL ----------------
        refined = sorted(refined, key=lambda x: x["final_score"], reverse=True)

        return {
            "success": True,
            "best_match": refined[0],
            "similar": refined
        }


# INIT
service = ImageSimilarityService()
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
import easyocr
from ultralytics import YOLO

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

        self.yolo_model = YOLO("yolov8s.pt")

        self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

        self.brand_texts = ["rolex watch", "omega watch", "seiko watch"]
        self.text_tokens = clip.tokenize(self.brand_texts).to(self.device)

        self.index = None
        self.metadata = []

    # ---------------- BASIC UTILS ----------------
    def preprocess_image(self, img):
        img = img.resize((512, 512))
        img = np.array(img)
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return Image.fromarray(img)

    def enhance_image(self, img):
        img = np.array(img)
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        img = cv2.filter2D(img, -1, kernel)
        img = cv2.convertScaleAbs(img, alpha=1.3, beta=10)
        return Image.fromarray(img)

    def is_blur(self, img):
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < 50

    def center_crop(self, img):
        w, h = img.size
        return img.crop((w//4, h//4, w*3//4, h*3//4))

    def is_already_zoomed(self, original_img, cropped_img):
        ow, oh = original_img.size
        cw, ch = cropped_img.size
        return (cw * ch) / (ow * oh) > 0.7

    # ---------------- DETECTION ----------------
    def detect_watch(self, img):
        try:
            results = self.yolo_model(np.array(img), verbose=False)

            best_box, best_conf = None, 0

            for r in results:
                if r.boxes is None:
                    continue

                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):

                    if int(cls) != 74:  # clock
                        continue

                    if conf < 0.3:
                        continue

                    if conf > best_conf:
                        best_conf = conf
                        best_box = box

            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box)
                return img.crop((x1, y1, x2, y2))

            return self.center_crop(img)

        except:
            return img

    def detect_dial(self, img):
        w, h = img.size
        cx, cy = w//2, h//2
        size = int(min(w, h) * 0.5)
        return img.crop((cx-size//2, cy-size//2, cx+size//2, cy+size//2))

    # ---------------- OCR & BRAND ----------------
    def extract_text(self, img):
        try:
            result = self.ocr_reader.readtext(np.array(img))
            return " ".join([r[1].lower() for r in result])
        except:
            return ""

    def detect_brand(self, img):
        try:
            image = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(self.text_tokens)
                sim = (image_features @ text_features.T).softmax(dim=-1)
            return self.brand_texts[sim.argmax().item()].split()[0]
        except:
            return None

    # ---------------- EMBEDDING ----------------
    def get_clip_embedding(self, img):
        img = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(img)
        emb = emb.cpu().numpy()[0]
        return emb / np.linalg.norm(emb)

    def get_cnn_embedding(self, img):
        img = img.resize((224,224))
        img = torch.tensor(np.array(img)).permute(2,0,1).float()/255.0
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.cnn_model(img)
        return emb.cpu().numpy()[0]

    # ---------------- BUILD ----------------
    def build(self, products):

        embeddings, meta = [], []

        for i in range(0, len(products), 8):
            batch = products[i:i+8]
            results = asyncio.run(self.download_batch(batch))

            for img, p in results:
                if img is None:
                    continue

                img = self.preprocess_image(img)

                if self.is_blur(img):
                    img = self.enhance_image(img)

                watch = self.detect_watch(img)

                dial = watch if self.is_already_zoomed(img, watch) else self.detect_dial(watch)
                center = self.center_crop(img)

                clip = (self.get_clip_embedding(watch) +
                        self.get_clip_embedding(dial) +
                        self.get_clip_embedding(center)) / 3

                cnn = (self.get_cnn_embedding(watch) +
                       self.get_cnn_embedding(dial)) / 2

                emb = np.concatenate([clip*0.6, cnn*0.4])

                p["ocr"] = self.extract_text(dial)
                p["brand"] = self.detect_brand(watch)

                embeddings.append(emb)
                meta.append(p)

            gc.collect()

        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
        self.index.add(embeddings)

        self.metadata = meta

        faiss.write_index(self.index, "crono1.faiss")
        json.dump(meta, open("cron_data1.json","w"))
    def load(self):
        self.index = faiss.read_index("crono1.faiss")
        self.metadata = json.load(open("cron_data1.json"))

    async def download_batch(self, batch):
        async with aiohttp.ClientSession() as session:
            tasks = [self.async_download(session, p) for p in batch]
            return await asyncio.gather(*tasks)

    async def async_download(self, session, p):
        try:
            print("images ",p["image"])
            async with session.get(p["image"], timeout=10) as resp:
                if resp.status != 200:
                    return None, p
                data = await resp.read()
                return Image.open(BytesIO(data)).convert("RGB"), p
        except:
            return None, p

    def download_image(self, url):
        try:
            r = session.get(url, timeout=10)
            if r.status_code != 200:
                return None
            return Image.open(BytesIO(r.content)).convert("RGB")
        except:
            return None

    # ---------------- SEARCH ----------------
    def search(self, url, top_k=5):

        img = self.download_image(url)
        if img is None:
            return {"success": False}

        img = self.preprocess_image(img)

        if self.is_blur(img):
            img = self.enhance_image(img)

        watch = self.detect_watch(img)
        dial = watch if self.is_already_zoomed(img, watch) else self.detect_dial(watch)
        center = self.center_crop(img)

        clip = (self.get_clip_embedding(watch) +
                self.get_clip_embedding(dial) +
                self.get_clip_embedding(center)) / 3

        cnn = (self.get_cnn_embedding(watch) +
               self.get_cnn_embedding(dial)) / 2

        query = np.concatenate([clip*0.6, cnn*0.4])

        text = self.extract_text(dial)
        brand = self.detect_brand(watch)

        D, I = self.index.search(query.reshape(1,-1), 50)

        results = []

        for score, idx in zip(D[0], I[0]):
            m = self.metadata[idx]

            final = float(score)

            if text and m.get("ocr") and any(w in m["ocr"] for w in text.split()):
                final += 0.2

            if brand and m.get("brand") == brand:
                final += 0.3

            results.append({
                "product_id": m["product_id"],
                "image": m["image"],
                "score": final
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return {
            "success": True,
            "best_match": results[0],
            "similar": results
        }


# INIT
service = ImageSimilarityService()
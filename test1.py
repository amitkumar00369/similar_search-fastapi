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
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, embedding_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return F.normalize(self.fc(x), dim=1)


class ImageSimilarityService:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # CLIP model
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

        # CNN (re-ranking)
        self.cnn_model = SimpleCNN().to(self.device)
        self.cnn_model.eval()

        self.index = None
        self.metadata = []

        # 🔥 Known brands (auto detection)
        self.brand_list = ["Rolex", "Omega", "Casio", "Seiko", "Titan", "Fossil"]

    # ----------------------------
    # Download Image
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
            return None

    # ----------------------------
    # CLIP Embeddings
    # ----------------------------
    def get_batch_embeddings(self, images):
        images = [img for img in images if img is not None]
        if len(images) == 0:
            return np.array([])

        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)

        with torch.no_grad():
            embeddings = self.model.encode_image(batch)

        embeddings = embeddings.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings.astype("float32")

    # ----------------------------
    # CNN Embedding (for rerank)
    # ----------------------------
    def get_cnn_embedding(self, image):
        img = image.resize((224, 224))
        img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.cnn_model(img)

        return emb.cpu().numpy()[0]

    # ----------------------------
    # Auto Brand Detection (CLIP)
    # ----------------------------
    def detect_brand(self, image):
        inputs = self.preprocess(image).unsqueeze(0).to(self.device)

        text = clip.tokenize(self.brand_list).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(inputs)
            text_features = self.model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).softmax(dim=-1)

        best_idx = similarity.argmax().item()
        return self.brand_list[best_idx]

    # ----------------------------
    # Build Embeddings (Dataset Ready)
    # ----------------------------
    def build_embeddings(self, products, batch_size=64):

        all_embeddings = []
        final_metadata = []

        for i in range(0, len(products), batch_size):

            batch_products = products[i:i+batch_size]

            with ThreadPoolExecutor(max_workers=50) as executor:
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
                meta = batch_meta[j]

                # 🔥 ensure structure
                meta.setdefault("brand", "unknown")
                meta.setdefault("model", "unknown")

                all_embeddings.append(batch_emb[j])
                final_metadata.append(meta)

            del images, batch_imgs, batch_meta, batch_emb
            gc.collect()

        embeddings_array = np.array(all_embeddings).astype("float32")

        dimension = embeddings_array.shape[1]

        # 🔥 IVF index
        quantizer = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_INNER_PRODUCT)

        self.index.train(embeddings_array)
        self.index.add(embeddings_array)
        self.index.nprobe = 10

        self.metadata = final_metadata

    # ----------------------------
    # Search (Hybrid)
    # ----------------------------
    def search(self, image_url, product_ids=None, top_k=5):

        if self.index is None:
            raise Exception("Index not built!")

        img = self.download_image(image_url)

        if img is None:
            return {"success": False}

        # 🔥 CLIP embedding
        emb = self.get_batch_embeddings([img])
        if len(emb) == 0:
            return {"success": False}

        query = emb[0]

        # 🔥 Brand detection
        detected_brand = self.detect_brand(img)

        # 🔥 FAISS search
        D, I = self.index.search(query.reshape(1, -1), 100)

        results = []

        for score, idx in zip(D[0], I[0]):
            if idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]

            # 🔥 Brand filter boost
            brand_boost = 0.05 if meta.get("brand") == detected_brand else 0

            results.append({
                "id": meta["id"],
                "product_id": meta["product_id"],
                "image": meta["image"],
                "score": float(score),
                "brand": meta.get("brand"),
                "final_score": float(score + brand_boost)
            })

        # 🔥 Remove duplicates
        seen = set()
        unique = []
        for r in results:
            if r["product_id"] not in seen:
                unique.append(r)
                seen.add(r["product_id"])

        results = unique

        # 🔥 CNN re-ranking
        query_cnn = self.get_cnn_embedding(img)

        for r in results:
            ref_img = self.download_image(r["image"])
            if ref_img is None:
                continue

            ref_emb = self.get_cnn_embedding(ref_img)

            sim = np.dot(query_cnn, ref_emb)
            r["final_score"] += float(sim * 0.1)

        results = sorted(results, key=lambda x: x["final_score"], reverse=True)

        best = results[0] if results else None

        return {
            "success": True,
            "detected_brand": detected_brand,
            "best_match": best,
            "similar_matches": results[1:top_k],
            "total_results": len(results)
        }


# instance
similar = ImageSimilarityService()
import pandas as pd
import json

data = pd.read_excel("products (11) (1).xlsx")

all_rolex = []

for i, row in enumerate(data.to_dict(orient="records"), start=1):
    val = {
        "id": row.get("sno"),
        "product_id": str(row.get("reference")),
        "image": row.get("product_image_url"),
        "colour": row.get("dial_color"),
        "material": row.get("case_material"),
        "diameter": row.get("case_diameter"),
        "price": row.get("price_usd") or 0,
        "model": row.get("model")
    }
    all_rolex.append(val)

# Save output
with open("rolex_data.json", "w", encoding="utf-8") as f:
    json.dump(all_rolex, f, indent=4, ensure_ascii=False)

print("✅ Done! Generated", len(all_rolex), "products")
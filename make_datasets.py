import pandas as pd
import json

data = pd.read_csv("3_ref-datasets.csv")
print(data.columns)
all_rolex = []

for i, row in enumerate(data.to_dict(orient="records"), start=1):
    if row.get("ref_no")==126200:
        continue
    else:
        
        val = {
            "id": row.get("id"),
            "product_id": str(row.get("ref_no")),
            "image": row.get("image"),
            # "colour": row.get("dial_color"),
            # "material": row.get("case_material"),
            # "diameter": row.get("case_diameter"),
            # "price": row.get("price_usd") or 0,
            "image_id": row.get("image_id") or "rolex"
        }
    all_rolex.append(val)

# Save output
with open("rolex_clean_data_3_ref.json", "w", encoding="utf-8") as f:
    json.dump(all_rolex, f, indent=4, ensure_ascii=False)

print("✅ Done! Generated", len(all_rolex), "products")
from te import data
import json

print(data.keys())
results = data["results"]

final_data = []

for idx, item in enumerate(results, start=41):
    title = item.get("title", "")
    rmc = item.get("rmc", "")
    alt = item.get("alt", "")
    price = item.get("price", "")

    # 🔥 Build Image URL
    image_url = f"https://media.rolex.com/image/upload/q_auto/f_auto/t_v7-cover-majesty-landscape/c_limit,w_1920/v1/catalogue/2025/upright-c/{rmc}"

    # 🔥 Extract product_id from rmc
    product_id = ""
    if rmc:
        try:
            product_id = rmc[1:].split("-")[0]   # remove 'm' and take before '-'
        except:
            pass

    # 🔥 Parse ALT
    parts = [x.strip() for x in alt.split(",")]

    diameter = ""
    material = ""
    colour = ""

    try:
        diameter = parts[2].upper().replace(" ", "")
    except:
        pass

    try:
        material = parts[3]
    except:
        pass

    try:
        if "Dial" in parts[4]:
            colour = parts[4].split(":")[-1].strip()
    except:
        pass

    # 🔥 Final object
    obj = {
        "id": idx,
        "product_id": product_id,   # ✅ FIXED HERE
        "image": image_url,
        "colour": colour,
        "material": material,
        "diameter": diameter,
        "price": price
    }

    final_data.append(obj)

# Save output
with open("rolex.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=4, ensure_ascii=False)

print("✅ Done! Generated", len(final_data), "products")
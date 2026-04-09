import requests
from bs4 import BeautifulSoup
import json
import time

BASE_URL = "https://www.ethoswatches.com/brands.html"

headers = {
    "User-Agent": "Mozilla/5.0"
}

def scrape_ethos(max_pages=60):
    all_products = []
    id_counter = 1

    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}?p={page}&product_list_limit=99"

        print(f"\n🚀 Scraping page {page}")

        resp = requests.get(url, headers=headers, timeout=10)

        # if resp.status_code != 200:
        #     print("❌ Failed page:", page)
        #     continue

        soup = BeautifulSoup(resp.text, "html.parser")

        products = soup.select("li.product-item")

        print(f"Found {len(products)} items")

        if not products:
            break

        for p in products:
            try:
                # image
                img_tag = p.select_one("img")
                image = img_tag.get("src") if img_tag else None

                # product name
                name_tag = p.select_one(".product-item-link")
                name = name_tag.text.strip() if name_tag else ""

                # price
                price_tag = p.select_one(".price")
                price = price_tag.text.strip() if price_tag else ""

                if not image:
                    continue

                all_products.append({
                    "id": id_counter,
                    "product_id": name,
                    "image": image,
                    "price": price
                })

                id_counter += 1

            except Exception as e:
                print("❌ Error:", e)

        time.sleep(1)  # avoid blocking

    # save file
    with open("ethos_dataset.json", "w") as f:
        json.dump(all_products, f, indent=2)

    print("\n✅ DONE")
    print(f"Total products: {len(all_products)}")


if __name__ == "__main__":
    scrape_ethos(max_pages=60)
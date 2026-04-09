from playwright.sync_api import sync_playwright
import json
import time

BASE_URL = "https://www.omegawatches.com/watchfinder"

def scrape_omega(max_pages=20):
    all_products = []
    product_id_counter = 1

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for page_no in range(1, max_pages + 1):
            url = f"{BASE_URL}?p={page_no}"
            print(f"\n🚀 Scraping page {page_no}: {url}")

            page.goto(url, timeout=60000)
            page.wait_for_timeout(3000)

            # scroll to load all items
            page.mouse.wheel(0, 5000)
            page.wait_for_timeout(2000)

            cards = page.query_selector_all(".product-item")

            print(f"Found {len(cards)} products")

            if len(cards) == 0:
                break

            for card in cards:
                try:
                    # image
                    img_el = card.query_selector("img")
                    image = img_el.get_attribute("src") if img_el else None

                    # name
                    name_el = card.query_selector(".product-item-name")
                    name = name_el.inner_text().strip() if name_el else ""

                    # details (optional)
                    desc_el = card.query_selector(".product-item-attribute")
                    desc = desc_el.inner_text().strip() if desc_el else ""

                    if not image:
                        continue

                    # 🔥 clean product_id
                    product_id = name.replace("\n", " ").strip()

                    all_products.append({
                        "id": product_id_counter,
                        "product_id": product_id,
                        "image": image,
                        "description": desc
                    })

                    product_id_counter += 1

                except Exception as e:
                    print("❌ Error parsing:", e)

        browser.close()

    # 🔥 save dataset
    with open("omega_dataset.json", "w") as f:
        json.dump(all_products, f, indent=2)

    print("\n✅ DONE")
    print(f"Total products scraped: {len(all_products)}")


# # if __name__ == "__main__":
#     scrape_omega(max_pages=30)

print(scrape_omega(25))
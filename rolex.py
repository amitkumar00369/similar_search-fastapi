from playwright.sync_api import sync_playwright
import json
import time

URL = "https://www.rolex.com/watches/find-rolex?group=0"

def auto_scroll(page):
    previous_height = 0

    while True:
        page.mouse.wheel(0, 5000)
        page.wait_for_timeout(2000)

        current_height = page.evaluate("document.body.scrollHeight")

        if current_height == previous_height:
            break

        previous_height = current_height


def scrape_rolex():
    data = []
    id_counter = 1

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("🚀 Opening Rolex site...")
        page.goto(URL, timeout=60000)
        print(page)

        page.wait_for_timeout(5000)

        print("📜 Scrolling to load all watches...")
        auto_scroll(page)

        print("🔍 Extracting data...")

        cards = page.query_selector_all("li")  # adjust selector if needed

        for card in cards:
            try:
                # image
                img_el = card.query_selector("img")
                image = img_el.get_attribute("src") if img_el else None

                # title
                title_el = card.query_selector("h3")
                title = title_el.inner_text().strip() if title_el else ""

                # description
                desc_el = card.query_selector("p")
                desc = desc_el.inner_text().strip() if desc_el else ""

                if not image or "rolex" not in image:
                    continue

                data.append({
                    "id": id_counter,
                    "product_id": title,
                    "image": image,
                    "description": desc
                })

                id_counter += 1

            except Exception as e:
                print("❌ Error:", e)

        browser.close()

    with open("rolex_dataset.json", "w") as f:
        json.dump(data, f, indent=2)

    print("✅ DONE")
    print(f"Total scraped: {len(data)}")


print(scrape_rolex())
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import os
import requests

# ---------------- CONFIG ----------------
BASE_URL = "https://www.chrono24.com/search/index.htm?dosearch=true&query=Rolex+126621&showpage={}"
TOTAL_PAGES = 6
SAVE_DIR = "chrono_images"

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- DRIVER ----------------
options = Options()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)

all_images = set()

# ---------------- SCRAPER ----------------
for page in range(1, TOTAL_PAGES + 1):
    print(f"\n🔄 Scraping Page {page}")

    driver.get(BASE_URL.format(page))
    time.sleep(5)

    # scroll for lazy loading
    for _ in range(5):
        driver.execute_script("window.scrollBy(0, 1200);")
        time.sleep(2)

    # 👉 ONLY FIRST IMAGE PER LISTING
    cards = driver.find_elements(By.XPATH, "//div[contains(@class,'listing-item')]")

    print("Listings found:", len(cards))

    for idx, card in enumerate(cards):
        try:
            img = card.find_element(By.XPATH, ".//div[contains(@class,'watch-image')]//img")

            src = img.get_attribute("data-lazy-src") or img.get_attribute("src")

            if src and "chrono24" in src:
                all_images.add(src)

        except Exception as e:
            continue

driver.quit()

print("\n✅ Total unique images:", len(all_images))

# ---------------- DOWNLOAD ----------------
print("\n⬇️ Downloading images...")

for i, url in enumerate(all_images):
    try:
        response = requests.get(url, timeout=10)
        path = os.path.join(SAVE_DIR, f"watch_{i}.jpg")

        with open(path, "wb") as f:
            f.write(response.content)

        print(f"Saved {i}")

    except:
        print(f"❌ Failed {i}")

print("\n🎉 DONE")
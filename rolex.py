# from playwright.sync_api import sync_playwright
# import json
# import time

# URL = "https://www.rolex.com/watches/find-rolex?group=0"

# def auto_scroll(page):
#     previous_height = 0

#     while True:
#         page.mouse.wheel(0, 5000)
#         page.wait_for_timeout(2000)

#         current_height = page.evaluate("document.body.scrollHeight")

#         if current_height == previous_height:
#             break

#         previous_height = current_height


# def scrape_rolex():
#     data = []
#     id_counter = 1

#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=True)
#         page = browser.new_page()

#         print("🚀 Opening Rolex site...")
#         page.goto(URL, timeout=60000)
#         print(page)

#         page.wait_for_timeout(5000)

#         print("📜 Scrolling to load all watches...")
#         auto_scroll(page)

#         print("🔍 Extracting data...")

#         cards = page.query_selector_all("li")  # adjust selector if needed

#         for card in cards:
#             try:
#                 # image
#                 img_el = card.query_selector("img")
#                 image = img_el.get_attribute("src") if img_el else None

#                 # title
#                 title_el = card.query_selector("h3")
#                 title = title_el.inner_text().strip() if title_el else ""

#                 # description
#                 desc_el = card.query_selector("p")
#                 desc = desc_el.inner_text().strip() if desc_el else ""

#                 if not image or "rolex" not in image:
#                     continue

#                 data.append({
#                     "id": id_counter,
#                     "product_id": title,
#                     "image": image,
#                     "description": desc
#                 })

#                 id_counter += 1

#             except Exception as e:
#                 print("❌ Error:", e)

#         browser.close()

#     with open("rolex_dataset.json", "w") as f:
#         json.dump(data, f, indent=2)

#     print("✅ DONE")
#     print(f"Total scraped: {len(data)}")


# print(scrape_rolex())

# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.options import Options
# import time

# # Setup
# options = Options()
# options.add_argument("--start-maximized")

# driver = webdriver.Chrome(options=options)

# url = "https://www.rolex.com/en-in/watches/find-rolex"
# driver.get(url)

# time.sleep(5)  # wait initial load

# # 🔥 Scroll multiple times to load all images
# last_height = driver.execute_script("return document.body.scrollHeight")

# for _ in range(100):   # increase if more images
#     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#     time.sleep(5)

#     new_height = driver.execute_script("return document.body.scrollHeight")
#     if new_height == last_height:
#         break
#     last_height = new_height

# # 🔥 Extract images
# images = driver.find_elements(By.TAG_NAME, "img")

# img_urls = []
# for img in images:
#     src = img.get_attribute("src")
#     if src and "rolex.com" in src:
#         img_urls.append(src)

# # Remove duplicates
# img_urls = list(set(img_urls))

# print(f"Total Images: {len(img_urls)}")

# # for url in img_urls:
# #     print(url)

# driver.quit()

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

options = Options()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)
driver.get("https://www.rolex.com/en-in/watches/find-rolex")

time.sleep(5)

img_urls = set()

for i in range(100):   # increase for more scroll
    images = driver.find_elements(By.TAG_NAME, "img")

    for img in images:
        srcset = img.get_attribute("srcset")
        src = img.get_attribute("src")

        # 🔥 get HIGH RES image
        if srcset:
            high_res = srcset.split(",")[-1].split(" ")[0]
            img_urls.add(high_res)
        elif src:
            if "rolex.com" in src:
                img_urls.add(src)
    print(f"Scroll {i} → Images collected: {len(img_urls)}")

    # scroll
    driver.execute_script("window.scrollBy(0, 2000);")
    time.sleep(3)

print("Total images:", len(img_urls))

# for url in img_urls:
#     print(url)

driver.quit()
import time
from test import similar

def search(url: str,id:list):
    print("urls", url)

    similar.load_embeddings()   # 🔴 very important

    result = similar.search(url,id)

    return result


# input_url = "https://d5ee3ksv7elb9.cloudfront.net/product/0eb1f10b-53ad-4e68-ac8a-0ee268775875.png"
# input_url = "https://d5ee3ksv7elb9.cloudfront.net/product/5ad23231-c248-4001-acc0-c905e0c94fef.png"
# input_url = "https://d5ee3ksv7elb9.cloudfront.net/watch_media/1755656437_6620.jpg"


# print("start", time.time())
# print("results in output", search(input_url))
# print("end", time.time())

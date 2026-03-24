import requests
from test import similar

# products =[
#     {
#       "id": 1,
#       "product_id": 26,
#       "image": "https://d5ee3ksv7elb9.cloudfront.net/watch_media/1755656437_7338.jpg"
#     },
#   ]

url = "https://i-chrono.com/api/Import-products-vector"

products = requests.get(url).json()

# input_url = "https://d5ee3ksv7elb9.cloudfront.net/product/0eb1f10b-53ad-4e68-ac8a-0ee268775875.png"
similar.build_embeddings(products)
similar.save()

import uvicorn
from fastapi import FastAPI, Query
from typing import List
from test2 import ImageSimilarityService
# from result import search
from pydantic import BaseModel

app = FastAPI()

service = ImageSimilarityService()
service.load()

class SearchRequest(BaseModel):
    url: str
    product_ids: List[int] = []

@app.post("/search")
def search_api(data: SearchRequest):
    try:
        result = service.search(data.url, data.product_ids)
        return result
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
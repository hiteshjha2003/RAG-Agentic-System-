from fastapi import FastAPI, Query
import uvicorn
from typing import List, Dict, Any  
from search_engine import SearchEngine
from tqdm import tqdm
import os

dataset_dir = './search_engine/corpus'


app = FastAPI(
    title="Hybrid Search Engine API",
    description="Provides search functionality using a pre-initialized HybridSearchEngine.",
    version="1.0.0",
)


search_engine=None

@app.on_event("startup")
async def startup_event():
    global search_engine
    print("Initializing SearchEngine...")
    search_engine = SearchEngine(dataset_dir, embed_model_name='vidore/colqwen2-v1.0')



@app.get(
    "/search",
    summary="Perform a search query.",
    description="Executes a search using the initialized SearchEngine and returns the results.",
    response_model=List[List[Dict[str, Any]]]  
)
async def search(queries: List[str] = Query(...)):
    
    results_batch = search_engine.batch_search(queries)
    results_batch = [[dict(idx=idx,image_file=os.path.join(f'./search_engine/corpus/img',file)) for idx,file in enumerate(query_results)] for query_results in results_batch]
    return results_batch

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8002)

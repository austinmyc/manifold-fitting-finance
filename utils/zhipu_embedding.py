from zhipuai import ZhipuAI
from dotenv import load_dotenv
import os

load_dotenv()

def get_zhipu_embedding(model = "embedding-3", input : list = []):
    if not input:
        raise
    client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY")) 
    response = client.embeddings.create(
        model=model, 
        input=input,
    )
    return response
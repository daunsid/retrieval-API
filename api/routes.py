from typing import List, Optional


from information_retrieval import config
from information_retrieval.utils.utils import load_embeddings
from information_retrieval.models.predict import drugs_information_retrieval
from .model import get_DI

from api.schema import   DrugInfo, RequestID
from fastapi import (
    HTTPException, status,
    APIRouter
)

with open('features.names', 'r') as fn:
    fields = fn.readlines()
    fields = [line.strip() for line in fields]
    fn.close

router = APIRouter()

@router.get('/home')
async def index():
    return {"Information Retrieval":"Search for information about drugs"}

@router.post("/drug_information", response_model= List[DrugInfo])
def get_drug_information(request:RequestID, ):

    text = request.query
    output = drugs_information_retrieval(text, load_embeddings(config['PATHS']['EMB_PATH']))
    idx, _ = output['corpus_id'], output['score']
    drug_info = get_DI(idx).split('\n')
    results = {field:info
            for field, info in zip(fields, drug_info)
            }
    results['side_effects'] = results['side_effects'].split(',')
    results['brand_names'] = results['brand_names'].split(',')
    results['related_drugs'] = results['related_drugs'].split(',')
    results['drug_classes'] = results['drug_classes'].split(',')

    return results
from sentence_transformers import util

from ..utils.utils import get_embeddings, get_tokenizer
from ..utils.datasets import to_string
from information_retrieval import config

def drugs_information_retrieval(texts, embeddings):
    inputs = to_string(texts)
    inputs = get_tokenizer(texts, config['PATHS']['MODEL_PATH'])
    
    prediction = get_embeddings(inputs).detach().cpu()
    #outputs = F.cosine_similarity(prediction, embeddings)
    embeddings = embeddings.detach().cpu()
    outputs = util.semantic_search(prediction, embeddings, top_k=1)[0][0]
    
    return outputs
    
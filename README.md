# DrugInformationRetrieval-API
**Semantic-search inference API** to retrive information based on query and different criteria: drug name, side effects, related drug, generic name, 

I used `sentence-transformers/all-MiniLM-L6-v2` model two distict steps

* For extracting feature from the text data

The `/data/raw` and the `/notebooks` directories are not necessary for following along: they are used in the exploratory and cleaning stages.
The `/information_retrieval` directory contains an important script 
* `predict.py:`: It contains the needed function to retrieve information with the transformer model
* `transformer.py`: it implements the the transformer which is the `Encoder` class that subclasses the `nn.Module` class.
* `datasets.py`: Also contain among other things the `InformationRetrieval` class which extends the `torch.utils.data.Dataset` for building a custom dataset class.

The application is located in the `/api` directory in `main.py`. I also implemented test scripts `test.py` for testing the endpoints

Below is a high-level overview of the API:
![alt text] 
Its contains the following routes:




To check the API is doc, you can open http://localhost:8000/docs: 
it would redirect you to the interactive interface where you can try the API from the browser. 
You get something like this:


Other ways to try the API request:

* Postman
* Curl cmd


Example:
```
curl --location --request POST 'http://127.0.0.1:8000/drug_information' \
--header 'Content-Type: application/json' \
--data-raw '{
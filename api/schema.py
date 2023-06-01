from pydantic import BaseModel

from typing import List, Optional

class DrugInfo(BaseModel):

    drug_name:str
    medical_condition: str
    side_effects: List[str]
    generic_name: str
    drug_classes:List[str]
    brand_names:List[str]
    activity:str
    rx_otc:str
    pregnancy_category:str
    csa:str
    alcohol:str
    related_drugs:List[str]

class RequestID(BaseModel):
    query:str
    focus:Optional[List[str]]

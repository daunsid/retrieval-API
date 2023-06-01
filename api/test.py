from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

test_exp = "Search for information about drugs"

def test_index():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "Information Retrieval":[test_exp]
    }
    print("TEST 1 PASSED")

def test_get_drug_information():
    response = client.post(
        "http://0.0.0.0:8080/drug_information",
        headers={
            "accept":"application/json",
            "Content-Type":"application/json"
        },
        json={
            "query":test_exp
        }
    )
    assert response.status_code == 200
    response = response.json()
    assert response["text"]
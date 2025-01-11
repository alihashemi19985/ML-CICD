import pytest 
from app import app

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_predict(client):
    response = client.post('/predict', json={"features":[15.3,396.9,4.98,5]})
    assert response.status_code == 200
    assert "prediction" in response.get_json()




import API
import requests

def test_api():
    response = requests.post('http://localhost:5001/predict', json={"user_id": 456221})
    assert response.status_code == 200
    predictions = response.json()
    assert isinstance(predictions, list)
    assert len(predictions) > 0

if __name__ == '__main__':
    test_api()

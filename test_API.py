import API
import requests

def test_api():
    response = requests.post('https://dashboardscoringcredit-4b3cd19d3108.herokuapp.com/', json={"user_id": 456221})
    assert response.status_code == 200
    predictions = response.json()
    assert isinstance(predictions, list)
    assert len(predictions) > 0

if __name__ == '__main__':
    test_api()

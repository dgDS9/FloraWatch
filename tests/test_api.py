from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Florawatch Foto",
    }


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "app": "api Florawatch Foto",
        "status": "running",
    }



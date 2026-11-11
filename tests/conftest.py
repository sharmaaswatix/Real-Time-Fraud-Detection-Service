import os

import pytest
from fastapi.testclient import TestClient

from ml.train import train


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("svc")
    summary = train(synthetic=True, out_dir=str(d), n_estimators=60)
    os.environ.update(MODEL_PATH=str(d / "model.joblib"), FRAUD_DB=str(d / "t.db"),
                      JWT_SECRET="test-secret-that-is-at-least-32-bytes-long", DEMO_USER="tester", DEMO_PASSWORD="pw")
    return summary


@pytest.fixture()
def client(env):
    from app.main import app
    return TestClient(app)


@pytest.fixture()
def auth(client):
    r = client.post("/auth/login", data={"username": "tester", "password": "pw"})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture()
def samples(client):
    return client.get("/samples").json()

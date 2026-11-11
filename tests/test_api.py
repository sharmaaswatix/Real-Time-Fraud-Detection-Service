def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_login_rejects_bad_credentials(client):
    assert client.post("/auth/login", data={"username": "tester", "password": "nope"}).status_code == 401


def test_login_returns_token(client):
    r = client.post("/auth/login", data={"username": "tester", "password": "pw"})
    assert r.status_code == 200 and r.json()["token_type"] == "bearer"


def test_predict_requires_auth(client, samples):
    assert client.post("/predict", json=samples["legit"][0]).status_code == 401
    assert client.post("/predict", json=samples["legit"][0], headers={"Authorization": "Bearer junk"}).status_code == 401


def test_predict_response_shape(client, auth, samples):
    r = client.post("/predict", json=samples["legit"][0], headers=auth)
    body = r.json()
    assert r.status_code == 200
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_label"] in {"low", "medium", "high"}
    assert body["flagged"] == (body["fraud_probability"] >= body["threshold"])


def test_fraud_scores_higher_than_legit(client, auth, samples):
    f = [client.post("/predict", json=s, headers=auth).json()["fraud_probability"] for s in samples["fraud"]]
    l = [client.post("/predict", json=s, headers=auth).json()["fraud_probability"] for s in samples["legit"]]
    assert sum(f) / len(f) > sum(l) / len(l)


def test_invalid_input_rejected(client, auth, samples):
    tx = dict(samples["legit"][0])
    missing = {k: v for k, v in tx.items() if k != "V5"}
    assert client.post("/predict", json=missing, headers=auth).status_code == 422
    assert client.post("/predict", json={**tx, "Amount": -5}, headers=auth).status_code == 422
    assert client.post("/predict", json={**tx, "Amount": "abc"}, headers=auth).status_code == 422
    assert client.post("/predict", json={**tx, "extra": 1}, headers=auth).status_code == 422


def test_predictions_are_logged(client, auth, samples):
    pid = client.post("/predict", json=samples["fraud"][0], headers=auth).json()["id"]
    rows = client.get("/transactions", headers=auth).json()
    assert any(r["id"] == pid and r["username"] == "tester" for r in rows)
    assert client.get("/transactions").status_code == 401


def test_flagged_filter(client, auth, samples):
    for s in samples["fraud"] + samples["legit"]:
        client.post("/predict", json=s, headers=auth)
    rows = client.get("/transactions?flagged_only=true", headers=auth).json()
    assert rows and all(r["flagged"] for r in rows)

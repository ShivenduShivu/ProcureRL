from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "ProcureRL" in response.json()["name"]


def test_reset_returns_session_id():
    response = client.post("/reset", json={"difficulty": "easy", "seed": 42})
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "observation" in data
    assert "seller_last_price" in data["observation"]


def test_step_after_reset():
    response = client.post("/reset", json={"difficulty": "easy", "seed": 1})
    session_id = response.json()["session_id"]
    step_response = client.post(
        "/step",
        json={
            "session_id": session_id,
            "action": "I offer $110. <BUYER_PRICE>110</BUYER_PRICE>",
        },
    )
    assert step_response.status_code == 200
    data = step_response.json()
    assert "reward" in data
    assert isinstance(data["reward"], float)
    assert "terminated" in data


def test_step_invalid_session_returns_404():
    response = client.post("/step", json={"session_id": "fake-id", "action": "test"})
    assert response.status_code == 404


def test_state_endpoint():
    response = client.post("/reset", json={"difficulty": "easy", "seed": 5})
    session_id = response.json()["session_id"]
    state_response = client.get(f"/state/{session_id}")
    assert state_response.status_code == 200
    assert "current_round" in state_response.json()


def test_full_episode_via_api():
    response = client.post("/reset", json={"difficulty": "easy", "seed": 7})
    session_id = response.json()["session_id"]
    obs = response.json()["observation"]
    done = False
    steps = 0
    while not done and steps < 15:
        price = obs["seller_last_price"] - 2
        step_response = client.post(
            "/step",
            json={
                "session_id": session_id,
                "action": f"I offer ${price:.2f}. <BUYER_PRICE>{price:.2f}</BUYER_PRICE>",
            },
        )
        data = step_response.json()
        obs = data["observation"]
        done = data["terminated"] or data["truncated"]
        steps += 1
    assert done

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_languages_declares_real_capabilities():
    langs = client.get("/languages").json()
    assert langs["fon"]["translation"] and not langs["fon"]["asr_windows"]
    assert not langs["guw"]["translation"]


def test_score_endpoint_ranks_and_explains():
    r = client.post("/score", json={"original": "Je chante pour toi ce soir", "source_lang": "fra",
                                    "target_lang": "eng",
                                    "candidates": ["I sing for you tonight", "Tonight I sing this song for you my love"]})
    data = r.json()
    assert r.status_code == 200 and data["candidates"][1]["rejected"]
    assert "I sing for you tonight" in data["explanation"].splitlines()[0]


def test_consent_requires_explicit_statement():
    assert client.post("/consents", json={"artist": "A", "statement": "ok"}).status_code == 400

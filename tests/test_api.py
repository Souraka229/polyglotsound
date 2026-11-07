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


def test_ui_builds_with_colab_matplotlib_backend():
    """Regression: Colab exports MPLBACKEND=module://matplotlib_inline... (absent from our venv)."""
    import subprocess
    import sys
    env = dict(__import__("os").environ, MPLBACKEND="module://matplotlib_inline.backend_inline")
    r = subprocess.run([sys.executable, "-c", "from app.ui.gradio_app import build; build(); print('ok')"],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-500:]

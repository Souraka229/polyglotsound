# PolyglotSound

Adaptateur musical multilingue : il fait chanter une chanson dans une autre langue en gardant son sens,
son rythme, sa mélodie et la voix (autorisée) de l'artiste. L'original n'est jamais modifié.

Langues : **fon, français, anglais** dans les 6 directions (fon→fr, fon→en, fr→fon, en→fon, fr→en, en→fr).
Le yorùbá est déclaré dans le registre des langues. Le goun et le dendi sont déclarés mais **non
traduisibles** (absents de NLLB-200).

> État : MVP fonctionnel en local sur CPU (Windows). Les mesures réelles sont dans
> [docs/BENCHMARK.md](docs/BENCHMARK.md) et les licences dans [docs/LICENSES.md](docs/LICENSES.md).
> **NLLB-200 et MMS-TTS sont sous licence non commerciale** : ce MVP est un outil de R&D et de démonstration.

## Ce qui est propre à PolyglotSound

Les modèles open source (Demucs, NLLB, Whisper, Seed-VC, AudioSeal) sont des briques. La valeur est dans
notre code, testé :

| Composant | Fichier | Rôle |
|---|---|---|
| Syllabation phonétique et tonale | `engines/syllable_engine/` | français (e muet **optionnel**, élisions, hiatus, rimes phonétiques), anglais (dictionnaire CMU : syllabes et accent lexical), fon (tons H/B/montant/descendant extraits des diacritiques, voyelles nasales, nasales syllabiques, digrammes gb/kp/xw/hw) |
| Gabarit musical | `engines/musical_constraints/template.py` | chaque ligne devient une liste de notes chantées (temps, durée, hauteur MIDI, force du temps), avec un schéma de rimes |
| Alignement syllabes → notes | `engines/musical_constraints/alignment.py` | programmation dynamique : 1 syllabe par note, mélisme, syllabes serrées, élision ; coût explicable |
| Musical Constraint Engine | `engines/musical_constraints/engine.py`, `criteria.py` | 9 critères notés de 0 à 1 et expliqués : sens, syllabes, rythme, accents (fr/en), **tons vs mélodie (fon)**, voyelles chantables, rimes, culture, fluidité. Contraintes dures et rejets motivés. Comparaison argumentée de 3 propositions |
| Préservation culturelle | `engines/cultural_adaptation/` | lexique éditable d'expressions à garder ou rendre (Mawu, vodún...) |
| Boucle génération / vérification | `app/services/adaptation.py` | NLLB n-best, LLM local contraint (Ollama), vérification, retour des erreurs mesurées au LLM, régénération |
| Sens par aller-retour | `engines/translation/semantic.py` | chrF aller + rétro-traduction, calibré sur le littéral, pénalité d'ajout de contenu |
| Workflow artiste et traçabilité | `app/core/registry.py` | consentements (révocables), œuvres (SHA-256), générations versionnées, validations, filigrane relié au registre |

Un critère qui ne peut pas être mesuré de façon fiable (tons non écrits, durées estimées, sens) est
**marqué** et envoyé en validation humaine. Il n'est jamais présenté comme vérifié.

## Pipeline

```
MP3/WAV/FLAC ─► FFmpeg ─► Demucs htdemucs ─► voix ─┬─► pYIN (F0) ─► notes
                                └► instrumental ───┼─► tempo + temps forts (correction de phase)
                                                   └─► Whisper small (fr/en) | saisie artiste (fon)
paroles corrigées ─► alignement ─► gabarits musicaux ─► NLLB n-best + LLM contraint
                                                       └► Musical Constraint Engine ─► 3 adaptations notées
choix de l'artiste ─► Studio : voix guide chantée ─┐
                   └► Auto (expérimental) : MMS-TTS + WORLD sur la mélodie originale ─┤
                                                         consentement vérifié ─► Seed-VC (timbre)
voix convertie ─► AudioSeal (voix + mix) ─► mixage sur l'instrumental original ─► WAV + MP3 + provenance.json
```

## Lancer sur Google Colab (GPU gratuit)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Souraka229/polyglotsound/blob/main/notebooks/PolyglotSound_Colab.ipynb)

1. Cliquez sur le bouton, puis *Execution > Modifier le type d'execution > GPU T4*.
2. Lancez les 4 cellules dans l'ordre : code, installation complète et tests (10 à 15 min),
   démo de bout en bout, studio artiste (lien public `gradio.live`).

Ou en une commande sur toute machine Linux :

```bash
git clone https://github.com/Souraka229/polyglotsound && cd polyglotsound
WITH_LLM=1 bash scripts/setup_colab.sh
.venv/bin/python -m app.ui.gradio_app --share --host 0.0.0.0
```

Sur GPU, Demucs, AudioSeal, MMS-TTS et Seed-VC tournent sur CUDA. NLLB et Whisper restent sur CPU
exprès : leur build CUDA demande une version de cuDNN différente de celle de PyTorch.

## Installation (Windows 10/11, CPU)

Prérequis : Python 3.11 (via [uv](https://docs.astral.sh/uv/)), Git, FFmpeg dans le PATH
(`winget install Gyan.FFmpeg`). Optionnel : [Ollama](https://ollama.com) avec `ollama pull llama3.2`.
Espace disque : environ 9 Go au total (environnements et modèles).

```powershell
cd polyglotsound
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe scripts\download_models.py      # NLLB int8, Demucs, AudioSeal, Whisper small, CMU
powershell -ExecutionPolicy Bypass -File scripts\setup_seed_vc.ps1   # voix chantée (environnement isolé)
```

## Utilisation

```powershell
.venv\Scripts\python.exe -m app.ui.gradio_app      # http://127.0.0.1:7860
.venv\Scripts\python.exe -m uvicorn app.api.main:app --port 8000   # API (docs : /docs)
```

Parcours artiste : 1. importer le morceau, 2. corriger ou saisir les paroles, 3. choisir parmi 3
adaptations notées, 4. donner son consentement puis chanter une voix guide (Studio) ou utiliser
le mode Auto expérimental, 5. mixer, filigraner et télécharger.

## Tests et benchmarks

```powershell
.venv\Scripts\python.exe -m pytest                 # tests rapides, sans modèles
.venv\Scripts\python.exe -m pytest -m models       # intégration sur les vrais modèles
.venv\Scripts\python.exe scripts\make_synthetic_song.py
.venv\Scripts\python.exe benchmarks\bench_synthetic.py
.venv\Scripts\python.exe benchmarks\bench_adaptation.py [--no-llm]
.venv\Scripts\python.exe benchmarks\bench_svc.py 25
```

## Structure

```
app/        api (FastAPI), ui (Gradio), core (audio, langues, registre), services (pipeline, adaptation), schemas
engines/    audio_separation, speech_recognition, forced_alignment, music_analysis, translation,
            cultural_adaptation, syllable_engine, musical_constraints, singing_voice_conversion,
            watermarking, audio_mixing
models/     poids téléchargés (non versionnés) + dictionnaire CMU (BSD)
datasets/   jeux de test, projets, registre SQLite
benchmarks/ scripts et résultats mesurés     third_party/  Seed-VC (GPL, isolé)
configs/    lexique culturel                 docs/  benchmark, licences
```

## Éthique

- Aucune conversion de voix sans consentement explicite enregistré (`ConsentError` sinon). Le consentement est révocable.
- Chaque version générée porte un filigrane AudioSeal dont le message de 16 bits renvoie à son entrée dans le registre (œuvre, artiste, version, modèles).
- Le filigrane prouve une origine **lorsqu'il est détectable**. Ce n'est pas une preuve juridique de consentement.

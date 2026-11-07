# Licences des dépendances et des modèles

Vérifiées le 2026-11-07 (métadonnées des paquets installés, pages GitHub et cartes Hugging Face).
Ce document n'est pas un avis juridique.

## Code (environnement principal)

| Composant | Version | Licence | Usage dans PolyglotSound |
|---|---|---|---|
| PyTorch / torchaudio | 2.5.1 (CPU) | BSD-3-Clause | calcul |
| Demucs | 4.0.1 | MIT | séparation voix / instrumental |
| faster-whisper | 1.0.3 | MIT | ASR fr / en / yo |
| CTranslate2 | 4.4.0 | MIT | inférence NLLB int8 |
| sentencepiece | 0.2.3 | Apache-2.0 | tokenisation NLLB |
| transformers | 4.44.2 | Apache-2.0 | MMS-TTS (mode Auto) |
| AudioSeal | 0.1.4 | MIT (code **et** poids) | filigrane |
| librosa | 0.10.2.post1 | ISC | tempo, F0 (pYIN), chroma |
| pyworld (WORLD) | 0.3.5 | MIT (pyworld) / BSD modifiée (WORLD) | imposition de mélodie, mode Auto |
| Gradio | 4.44.1 | Apache-2.0 | interface |
| FastAPI / Pydantic | 0.115.0 / 2.9.2 | MIT | API, schémas |
| FFmpeg (binaire système) | 9.0.2 gyan.dev full | GPL-3.0 (build "full") | décodage / MP3, appelé en sous-processus |

## Données

| Donnée | Licence | Remarque |
|---|---|---|
| CMU Pronouncing Dictionary (`models/cmudict/`) | BSD-2 (CMU) | fichier copié avec sa licence, lu par notre propre chargeur. Le paquet PyPI `cmudict` est **GPL-3.0+** : il a été désinstallé volontairement. |

## Modèles

| Modèle | Licence | Usage commercial |
|---|---|---|
| NLLB-200 distilled 600M (conversion CT2 int8 `JustFrederik/nllb-200-distilled-600M-ct2-int8`) | **CC-BY-NC-4.0** | **Non.** Meta précise aussi : « not released for production deployment ». |
| Meta MMS-TTS (`facebook/mms-tts-fon/fra/eng`) | **CC-BY-NC-4.0** | **Non** |
| Demucs htdemucs | MIT | oui |
| Whisper small (Systran/faster-whisper-small) | MIT | oui |
| AudioSeal 16 bits | MIT | oui |
| Seed-VC `DiT_seed_v2_uvit_whisper_base_f0_44k` | dépôt GPL-3.0, pas de licence distincte annoncée pour les poids | à clarifier avec l'auteur |
| Omnilingual ASR (non installé, voir limites) | Apache-2.0 | oui |

## Outils isolés

| Outil | Licence | Isolation |
|---|---|---|
| Seed-VC (`third_party/seed-vc`, commit de la branche main, dernier push 2025-04-20) | GPL-3.0 | environnement séparé `third_party/seed-vc/.venv` (torch 2.4.1), appelé en **sous-processus** : aucun import de code GPL dans PolyglotSound. |

## Conséquences pour un produit commercial

1. **Traduction** : NLLB-200 doit être remplacé ou licencié. Pistes à évaluer avec un benchmark fon : MADLAD-400 (Apache-2.0, vérifier la couverture du fon), un modèle fine-tuné sur des corpus fon libres (FFR, ALFFA), ou un accord avec Meta.
2. **Mode Auto** : MMS-TTS est non commercial. Il faut un TTS ou SVS fon sous licence permissive, ou entraîner le nôtre.
3. **Seed-VC** : l'isolation en processus séparé évite de lier du code GPL, mais distribuer Seed-VC impose d'en fournir le code source. La licence des poids doit être confirmée. Alternative : RVC (MIT), qui demande un entraînement par artiste sur GPU.
4. **FFmpeg** : le build "full" est GPL. Un build LGPL suffit pour nos besoins (décodage, MP3 via libmp3lame).

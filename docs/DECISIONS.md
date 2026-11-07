# Choix techniques argumentés

Contexte matériel mesuré : Intel i5-1135G7 (4 cœurs / 8 threads), 15,7 Go de RAM, Intel Iris Xe
(**pas de CUDA**), Windows 11, et environ 15 Go libres sur le disque C après nettoyage des caches
npm et pnpm. Tout doit donc tourner **sur CPU**, en local.

## Séparation : Demucs htdemucs (retenu) vs BS-RoFormer

- **BS-RoFormer** est meilleur en SDR vocal d'après la littérature (vainqueur du Sound Demixing
  Challenge 2023). Mais le dépôt `lucidrains/BS-RoFormer` ne fournit **aucun poids entraîné**. Les
  poids viennent de la communauté (ZFTurbo, `audio-separator`), sous des licences et provenances
  hétérogènes, et le modèle est bien plus lourd sur CPU.
- **Demucs htdemucs** (MIT, 80 Mo, maintenu par Meta jusqu'en 2023, intégré par pip) : mesuré
  ici à **26,1 dB de SDR vocal** sur notre refrain synthétique. C'est un signal facile, ce chiffre
  n'est pas comparable à MUSDB18.
- Décision : Demucs par défaut. BS-RoFormer est prévu comme backend optionnel
  (`separate(..., backend="bs-roformer")`) après benchmark sur de vrais morceaux autorisés.

## Reconnaissance vocale : faster-whisper (fr/en) + saisie artiste (fon)

- **Whisper ne prend pas en charge le fon.** C'est vérifié : le fon est absent de la liste des
  langues de Whisper.
- **Omnilingual ASR** (Apache-2.0, plus de 1600 langues) : `fon_Latn` est probablement couvert,
  ce qui reste à confirmer avec `supported_langs`. Mais il dépend de **fairseq2, sans support
  Windows natif** (« no plans to support it », WSL2 uniquement). Le CTC 300M pèse 1,3 Gio, et il
  est entraîné sur de la parole, pas du chant.
- **WhisperX** : son apport, l'alignement forcé wav2vec2, n'a pas de modèle fon. Pour le fr/en,
  faster-whisper fournit déjà des timestamps par mot.
- Décision : faster-whisper small int8 pour fr/en/yo. Pour le fon, l'artiste saisit les paroles,
  les segments chantés détectés servant de repères. Prochaine étape : Omnilingual ASR sous WSL2,
  benchmarké sur du chant fon autorisé, en **suggestion** à corriger, jamais comme vérité.

## Traduction : NLLB-200 distilled 600M en int8 (CTranslate2)

- Codes `fon_Latn`, `fra_Latn`, `eng_Latn` présents. Les 6 directions sont testées et
  fonctionnent. La sortie fon contient des tons écrits (ex. « Un yí wǎn nú we »), ce qui rend le
  critère tonal mesurable.
- Le format int8 CT2 (620 Mo au lieu de 2,5 Go) prend environ 1 s par phrase sur ce CPU.
- **Licence CC-BY-NC-4.0, « not for production »** : bloquant pour le commercial, voir LICENSES.md.
- La version 1.3B distillée serait plus fiable pour le fon mais deux fois plus lourde : candidate
  pour le prochain benchmark.

## LLM d'adaptation : Ollama local (llama3.2 3B par défaut)

- Déjà installé sur la machine. llama3.2 (2 Go) est rapide sur CPU, gemma4 (9,6 Go) trop lourd
  pour la RAM libre.
- Le LLM n'est **jamais utilisé vers le fon** : sa qualité en fon n'est pas fiable. Chaque
  proposition est vérifiée par le Musical Constraint Engine, et les erreurs mesurées lui sont
  renvoyées pour régénération.

## Voix chantée : Seed-VC (retenu pour le MVP) vs RVC vs Amphion/Vevo2

| | Seed-VC | RVC | Amphion / Vevo2 |
|---|---|---|---|
| Chant | modèle dédié f0-conditionné 44,1 kHz | oui (avec RMVPE) | oui (recherche) |
| Nouvelle voix | **zéro-shot** (référence de 1 à 30 s) | **entraînement par artiste** (10 min ou plus de voix, GPU) | zéro-shot |
| CPU | possible, lent | inférence oui, entraînement non réaliste | lourd, orienté GPU |
| Licence | GPL-3.0 | MIT | MIT (code), poids variables |
| Maintenance | dernier push 2025-04 | communautaire | active (recherche) |

Décision : Seed-VC, isolé dans son propre environnement et appelé en sous-processus.
- Les dépendances figées (torch 2.4, transformers 4.46) ne polluent pas l'environnement principal.
- Le code GPL n'est pas lié à notre code.

RVC reste l'option « qualité maximale » une fois un GPU disponible et l'artiste d'accord pour
enregistrer un corpus.

## Mode Auto : MMS-TTS + WORLD (expérimental)

Un TTS parlé ne chante pas. Nous synthétisons chaque mot avec MMS-TTS (qui existe en fon), nous
l'étirons sur les notes que lui attribue l'alignement syllabes-notes, et nous lui imposons la F0
**chantée originale** avec le vocodeur WORLD. Le résultat est un guide robotique mais juste et en
place, que Seed-VC convertit ensuite au timbre de l'artiste. Il est étiqueté expérimental.

## F0 : pYIN (librosa) plutôt que RMVPE pour l'analyse

pYIN ne demande aucun téléchargement. Mesuré : 100 % des trames à moins de 50 cents sur la voix
vraie, 95,8 % sur la voix séparée. Le réglage 16 kHz avec un pas de 20 ms est 7 fois plus rapide
que le réglage par défaut, à précision égale. RMVPE, plus robuste sur des voix réelles bruitées,
est déjà téléchargé par Seed-VC : c'est l'amélioration prévue.

## Filigrane : AudioSeal 16 bits

MIT, code et poids. Le message de 16 bits porte l'identifiant de génération du registre. La voix
**et** le mix final sont marqués : quand seule la voix l'était, la détection dans le mix tombait à
p = 0,82.

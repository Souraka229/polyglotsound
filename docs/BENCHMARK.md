# Benchmark (mesures réelles)

Machine : Intel i5-1135G7, 4 cœurs, 16 Go de RAM, **CPU seul**, Windows 11. Date : 2026-11-07.
Fichiers bruts : `benchmarks/results/*.json`. Chaque chiffre ci-dessous sort de ces fichiers.

> **Limite majeure** : faute de chanson fon dont l'artiste a donné son accord, l'audio de test est
> un **refrain synthétique**. La voix est produite par MMS-TTS fon puis calée sur une mélodie par
> WORLD, sur une instrumentation synthétique. Ce signal est bien plus facile qu'un vrai mix. Ces
> chiffres valident la mécanique, **pas** la qualité sur de vraies chansons. Prochaine étape :
> constituer `datasets/authorized/` avec des refrains autorisés.

## 1. Pipeline audio (`bench_synthetic.py`, 12 s d'audio)

| Mesure | Résultat |
|---|---|
| Séparation Demucs htdemucs : SDR voix / instrumental | **26,05 dB / 25,83 dB** (mix brut pris comme voix : -1,61 dB) |
| Séparation : temps CPU | 79,3 s pour 12 s d'audio |
| Tempo estimé / vrai | 95,7 / 96 BPM |
| Temps forts, F-mesure (±70 ms) | **0,0 avant correction, 0,973 après** (correction de phase par énergie basse fréquence) |
| F0 pYIN sur voix séparée : précision à 50 cents / rappel du voisement | **95,8 %** / 97,4 % |
| F0 pYIN sur voix vraie | 99,8 % / 91,2 % |
| F0, réglage 16 kHz / pas 20 ms au lieu de 22 kHz / 11,6 ms | même précision (100 %), **4,8 s au lieu de 35,2 s** |
| Tonalité | **erreur** : mi mineur estimé, do majeur attendu. Non utilisée dans les scores |

## 2. Filigrane AudioSeal

| Cas | p(détection) | message 16 bits correct |
|---|---|---|
| voix non marquée | 0,000 | - |
| voix marquée | 1,000 | oui |
| voix marquée + instrumental | 0,818 | oui |
| idem après MP3 128 kbit/s | 0,810 | oui |
| mix original non marqué (faux positif) | 0,181 | - |

SNR du marquage : 34,6 dB. Conséquence : le pipeline marque aussi le **mix final**, pas seulement la voix.

## 3. Adaptation sur les 6 directions (`bench_adaptation.py`, 3 lignes par direction)

« Écart » = distance entre le nombre de syllabes et le nombre de notes de la mélodie (0 = parfait).
Score total entre 0 et 1 donné par le Musical Constraint Engine.

| Direction | Écart littéral NLLB | Écart meilleure proposition (sans LLM) | Écart (avec llama3.2) | Score littéral, puis meilleure (avec LLM) |
|---|---|---|---|---|
| fon → fr | 2,33 | 1,00 | 1,00 | 0,787 puis 0,823 |
| fon → en | 2,33 | 2,00 | 1,67 | 0,785 puis 0,803 |
| fr → fon | 2,33 | 1,33 | 1,33 (LLM non utilisé vers le fon) | 0,781 puis 0,824 |
| en → fon | 2,67 | 0,67 | 0,67 (LLM non utilisé vers le fon) | 0,767 puis 0,846 |
| fr → en | 0,33 | 0,33 | 0,33 | 0,898 puis 0,898 |
| en → fr | 1,00 | 1,00 | **0,33** | 0,832 puis 0,868 |

Lecture :
- Le moteur réduit l'écart syllabique dans 4 directions sur 6, surtout grâce au choix parmi les
  n-best de NLLB. Le LLM local aide surtout vers le français et l'anglais.
- Temps par ligne : 3 s sans LLM, 9 à 80 s avec llama3.2 sur CPU (mesuré pendant d'autres calculs,
  donc pessimiste).

Défauts trouvés par ce benchmark :
- **Corrigés** : le LLM était appelé même désactivé et même vers le fon. Il ajoutait du contenu
  (« ...quand la nuit est calme ») qui obtenait 0,907. Le sens est maintenant calibré sur
  l'aller-retour du littéral, avec une pénalité d'ajout.
- **Non résolu** : « Mawu ni kpé we » donne « que dieu te suffise » plutôt que « que dieu soit avec
  toi », parce que la première tombe pile sur les notes. Le proxy chrF ne voit pas ce glissement
  de sens. Le critère « sens » est donc toujours soumis à l'artiste.
- Les sorties fon de NLLB (« d'alɔ ») et les lignes fon du jeu de test **ne sont pas validées**
  par un locuteur natif.

## 4. Vérification d'un faux avertissement

transformers signale que les poids `weight_g/weight_v` de MMS-TTS ne sont pas utilisés. Après
vérification, les poids reconstruits correspondent au fichier à **1,5e-8** près : l'avertissement
est sans objet.

## 5. Conversion de voix chantée (Seed-VC)

Voir `benchmarks/results/svc_seedvc_*.json` (mesure CPU en cours au moment de la rédaction).

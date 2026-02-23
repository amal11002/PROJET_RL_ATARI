# projet-rl-atari
### Agent intelligent avec Gymnasium — 8INF974
*Phase 1 — Environnement et Baseline*

---

##  Description du projet

Ce projet implémente et compare deux algorithmes d'apprentissage par renforcement sur le jeu Atari Othello via la librairie Gymnasium : Monte Carlo Tree Search (MCTS) et Deep Q-Learning (DQN). La Phase 1 couvre l'installation de l'environnement et l'établissement d'une baseline avec un agent aléatoire.

---

### Cloner ou télécharger le projet

Place le dossier `projet-rl-atari` sur ton Bureau ou à l'emplacement de ton choix. Ouvre un terminal dans ce dossier.

###  Créer l'environnement virtuel

```bash
python -m venv venv
```

###  Activer l'environnement virtuel


```bash
venv\Scripts\activate
```



###  Installer les dépendances

```bash
pip install gymnasium[atari] ale-py numpy
```

Cette commande installe :

- `gymnasium` — l'interface principale pour interagir avec les jeux Atari
- `ale-py` — le moteur Arcade Learning Environment qui contient les ROMs
- `numpy` — librairie de calcul numérique

---

##  Structure du projet

| Fichier | Description |
|---|---|
| `phase1_test.py` | Vérifie l'installation et liste les environnements disponibles |
| `phase1_baseline.py` | Agent aléatoire sur Othello — établit le score de référence |

---

##  Exécution — Phase 1

### Vérifier l'installation

Lance ce script en premier pour confirmer que tout est bien installé :

```bash
python phase1_test.py
```

Résultat attendu : affichage du nombre d'environnements Atari disponibles (104) et quelques exemples.

###  Lancer l'agent aléatoire (baseline)

```bash
python phase1_baseline.py
```

Une fenêtre s'ouvrira avec le jeu Othello. L'agent jouera 10 parties au hasard. À la fin, le terminal affichera :

- Le score de chaque partie
- Le score moyen — notre référence de comparaison pour MCTS et DQN
- Le score minimum et maximum


## Notes techniques

- **Espace d'actions :** 10 actions discrètes (placements sur la grille + action neutre)
- **Espace d'observation :** images RGB de 210 × 160 × 3 pixels — l'entrée brute que recevra le CNN en Phase 3
- Le warning `'accept-rom-license'` lors de l'installation est normal — les ROMs sont désormais incluses directement dans `ale-py 0.11+`

---

## Dépannage courant

- **`(venv)` n'apparaît pas :** relance la commande d'activation `venv\Scripts\activate`
- **Erreur `No module named gymnasium` :** vérifie que l'environnement virtuel est activé avant d'installer
- **La fenêtre du jeu ne s'ouvre pas :** assure-toi d'utiliser `render_mode='human'` dans le script
import gymnasium as gym
import ale_py
import numpy as np

# Enregistrer les environnements Atari
gym.register_envs(ale_py)

# Charger l'environnement Othello
env = gym.make("ALE/Othello-v5", render_mode="human")

print(f"Espace d'actions : {env.action_space}")
print(f"Nombre d'actions possibles : {env.action_space.n}")
print(f"Espace d'observation (forme de l'image) : {env.observation_space.shape}")

# Agent aléatoire sur 10 parties
nb_parties = 10
scores = []

for partie in range(nb_parties):
    observation, info = env.reset()
    score_total = 0
    terminé = False

    while not terminé:
        # Choisir une action au hasard parmi toutes les actions possibles
        action = env.action_space.sample()
        
        # Exécuter l'action dans le jeu
        observation, récompense, terminé, tronqué, info = env.step(action)
        score_total += récompense
        
        if tronqué:
            terminé = True

    scores.append(score_total)
    print(f"Partie {partie + 1} — Score : {score_total}")

print(f"\nScore moyen sur {nb_parties} parties : {np.mean(scores):.2f}")
print(f"Score minimum : {np.min(scores):.2f}")
print(f"Score maximum : {np.max(scores):.2f}")

env.close()
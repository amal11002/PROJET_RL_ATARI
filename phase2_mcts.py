import gymnasium as gym
import ale_py
import numpy as np
from copy import deepcopy

gym.register_envs(ale_py)


# NOEUD DE L'ARBRE MCTS

class Node:
    def __init__(self, parent=None, action=None):
        self.parent = parent        # Noeud parent
        self.action = action        # Action qui a mené à ce noeud
        self.children = []          # Noeuds enfants
        self.wins = 0               # Nombre de victoires simulées
        self.visits = 0             # Nombre de fois visité

    def is_fully_expanded(self, nb_actions):
        return len(self.children) == nb_actions

    def ucb1(self, exploration=1.41):
        """Formule UCB1 : équilibre exploration et exploitation"""
        if self.visits == 0:
            return float('inf')  # Priorité absolue aux noeuds jamais visités
        return (self.wins / self.visits) + exploration * np.sqrt(
            np.log(self.parent.visits) / self.visits
        )

# LES 4 ÉTAPES DE MCTS

def select(node):
    """Descendre dans l'arbre en choisissant le meilleur UCB1"""
    while node.children:
        node = max(node.children, key=lambda n: n.ucb1())
    return node

def expand(node, nb_actions):
    """Ajouter un enfant non encore exploré"""
    actions_essayées = [c.action for c in node.children]
    actions_disponibles = [a for a in range(nb_actions) if a not in actions_essayées]
    action = np.random.choice(actions_disponibles)
    enfant = Node(parent=node, action=action)
    node.children.append(enfant)
    return enfant

def simulate(env_snapshot, action, nb_actions):
    """Jouer aléatoirement depuis cet état jusqu'à la fin"""
    env_copy = deepcopy(env_snapshot)
    _, récompense, terminé, tronqué, _ = env_copy.step(action)
    
    while not terminé and not tronqué:
        action_aléatoire = env_copy.action_space.sample()
        _, récompense, terminé, tronqué, _ = env_copy.step(action_aléatoire)
    
    env_copy.close()
    return récompense

def backpropagate(node, résultat):
    """Remonter le résultat jusqu'à la racine"""
    while node:
        node.visits += 1
        if résultat > 0:
            node.wins += 1
        node = node.parent


# AGENT MCTS COMPLET

def choisir_action_mcts(env, nb_simulations=50):
    """Choisir la meilleure action via MCTS"""
    racine = Node()
    nb_actions = env.action_space.n

    for _ in range(nb_simulations):
        # 1. Select
        noeud = select(racine)

        # 2. Expand (si pas encore exploré complètement)
        if not noeud.is_fully_expanded(nb_actions):
            noeud = expand(noeud, nb_actions)

        # 3. Simulate
        résultat = simulate(env, noeud.action, nb_actions)

        # 4. Backpropagate
        backpropagate(noeud, résultat)

    # Choisir l'action la plus visitée (la plus fiable)
    meilleur = max(racine.children, key=lambda n: n.visits)
    return meilleur.action

# JOUER DES PARTIES AVEC MCTS

env = gym.make("ALE/Othello-v5", render_mode="human")

nb_parties = 5  # Moins que baseline car MCTS est plus lent
scores = []
nb_simulations = 50  # Nombre de simulations par coup

print(f"MCTS avec {nb_simulations} simulations par coup")
print("=" * 40)

for partie in range(nb_parties):
    observation, info = env.reset()
    score_total = 0
    terminé = False
    nb_coups = 0

    while not terminé:
        action = choisir_action_mcts(env, nb_simulations)
        observation, récompense, terminé, tronqué, info = env.step(action)
        score_total += récompense
        nb_coups += 1

        if tronqué:
            terminé = True

    scores.append(score_total)
    print(f"Partie {partie + 1} — Score : {score_total} — Coups joués : {nb_coups}")

print(f"\nScore moyen MCTS  : {np.mean(scores):.2f}")
print(f"Score moyen baseline (aléatoire) : -10.60")
print(f"Amélioration : {np.mean(scores) - (-10.60):.2f} points")

env.close()
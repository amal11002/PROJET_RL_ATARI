import gymnasium as gym
import ale_py
import numpy as np

gym.register_envs(ale_py)

# NOEUD DE L'ARBRE MCTS

class Node:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0

    def is_fully_expanded(self, nb_actions):
        return len(self.children) == nb_actions

    def ucb1(self, exploration=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration * np.sqrt(
            np.log(self.parent.visits) / self.visits
        )

# LES 4 ÉTAPES MCTS

def select(node):
    while node.children:
        node = max(node.children, key=lambda n: n.ucb1())
    return node

def expand(node, nb_actions):
    actions_essayées = [c.action for c in node.children]
    actions_disponibles = [a for a in range(nb_actions) if a not in actions_essayées]
    action = np.random.choice(actions_disponibles)
    enfant = Node(parent=node, action=action)
    node.children.append(enfant)
    return enfant

def simulate(env_base, état_sauvegardé, nb_actions, max_coups=5):
    """Profondeur limitée à 15 coups"""
    env_base.restore_state(état_sauvegardé)
    action = np.random.choice(nb_actions)
    _, récompense, terminé, tronqué, _ = env_base.step(action)

    coups = 0
    while not terminé and not tronqué and coups < max_coups:
        action_aléatoire = np.random.choice(nb_actions)
        _, récompense, terminé, tronqué, _ = env_base.step(action_aléatoire)
        coups += 1

    return récompense

def backpropagate(node, résultat):
    while node:
        node.visits += 1
        if résultat > 0:
            node.wins += 1
        node = node.parent

# AGENT MCTS COMPLET

def choisir_action_mcts(env, nb_simulations=30):
    racine = Node()
    nb_actions = env.action_space.n
    env_base = env.unwrapped
    état_actuel = env_base.clone_state()

    for _ in range(nb_simulations):
        noeud = select(racine)

        if not noeud.is_fully_expanded(nb_actions):
            noeud = expand(noeud, nb_actions)

        résultat = simulate(env_base, état_actuel, nb_actions)
        backpropagate(noeud, résultat)

    env_base.restore_state(état_actuel)

    if not racine.children:
        return env.action_space.sample()

    meilleur = max(racine.children, key=lambda n: n.visits)
    return meilleur.action

# JOUER DES PARTIES

env = gym.make("ALE/Othello-v5")

nb_parties = 5
nb_simulations = 10
scores = []

print(f"MCTS avec {nb_simulations} simulations par coup | Profondeur : 15")
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

        if tronqué:
            terminé = True

    scores.append(score_total)
    print(f"Partie {partie+1} — Score : {score_total} — Coups : {nb_coups}")
    print("-" * 40)

print(f"\nScore moyen MCTS     : {np.mean(scores):.2f}")
print(f"Score moyen baseline : -10.60")
print(f"Amélioration         : {np.mean(scores) - (-10.60):.2f} points")

env.close()
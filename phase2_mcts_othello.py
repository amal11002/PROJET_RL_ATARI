import numpy as np
import time

# OTHELLO EN PYTHON PUR


VIDE = 0
NOIR = 1
BLANC = 2

def créer_plateau():
    plateau = np.zeros((8, 8), dtype=int)
    plateau[3][3] = BLANC
    plateau[4][4] = BLANC
    plateau[3][4] = NOIR
    plateau[4][3] = NOIR
    return plateau

def adversaire(joueur):
    return BLANC if joueur == NOIR else NOIR

def est_valide(plateau, ligne, col, joueur):
    if plateau[ligne][col] != VIDE:
        return False
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    adv = adversaire(joueur)
    for dl, dc in directions:
        l, c = ligne + dl, col + dc
        if 0 <= l < 8 and 0 <= c < 8 and plateau[l][c] == adv:
            l, c = l + dl, c + dc
            while 0 <= l < 8 and 0 <= c < 8:
                if plateau[l][c] == joueur:
                    return True
                if plateau[l][c] == VIDE:
                    break
                l, c = l + dl, c + dc
    return False

def coups_valides(plateau, joueur):
    return [(l, c) for l in range(8) for c in range(8)
            if est_valide(plateau, l, c, joueur)]

def jouer_coup(plateau, ligne, col, joueur):
    nouveau = plateau.copy()
    nouveau[ligne][col] = joueur
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    adv = adversaire(joueur)
    for dl, dc in directions:
        à_retourner = []
        l, c = ligne + dl, col + dc
        while 0 <= l < 8 and 0 <= c < 8 and nouveau[l][c] == adv:
            à_retourner.append((l, c))
            l, c = l + dl, c + dc
        if 0 <= l < 8 and 0 <= c < 8 and nouveau[l][c] == joueur:
            for rl, rc in à_retourner:
                nouveau[rl][rc] = joueur
    return nouveau

def est_terminé(plateau):
    return len(coups_valides(plateau, NOIR)) == 0 and \
           len(coups_valides(plateau, BLANC)) == 0

def score_final(plateau):
    noir = np.sum(plateau == NOIR)
    blanc = np.sum(plateau == BLANC)
    if noir > blanc:
        return 1
    elif blanc > noir:
        return -1
    return 0

def afficher_plateau(plateau):
    symboles = {VIDE: '.', NOIR: '●', BLANC: '○'}
    print("  0 1 2 3 4 5 6 7")
    for i, ligne in enumerate(plateau):
        print(f"{i} {' '.join(symboles[c] for c in ligne)}")
    print(f"  ● NOIR: {np.sum(plateau == NOIR)}  ○ BLANC: {np.sum(plateau == BLANC)}")

# MCTS — 
# chaque nœud stocke son propre état du plateau

class Node:
    def __init__(self, plateau, joueur, parent=None, coup=None):
        self.plateau = plateau      # État du plateau à CE nœud
        self.joueur = joueur        # Qui joue à CE nœud
        self.parent = parent
        self.coup = coup
        self.children = []
        self.wins = 0
        self.visits = 0
        self.coups_non_explorés = coups_valides(plateau, joueur)

    def est_fully_expanded(self):
        return len(self.coups_non_explorés) == 0

    def ucb1(self, exploration=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration * np.sqrt(
            np.log(self.parent.visits) / self.visits
        )

def select(node):
    """Descendre jusqu'à un nœud non complètement exploré"""
    while node.est_fully_expanded() and node.children:
        node = max(node.children, key=lambda n: n.ucb1())
    return node

def expand(node):
    """Ajouter un enfant non encore exploré"""
    if not node.coups_non_explorés:
        return node
    coup = node.coups_non_explorés.pop(np.random.randint(len(node.coups_non_explorés)))
    nouveau_plateau = jouer_coup(node.plateau, coup[0], coup[1], node.joueur)
    enfant = Node(
        plateau=nouveau_plateau,
        joueur=adversaire(node.joueur),
        parent=node,
        coup=coup
    )
    node.children.append(enfant)
    return enfant

def simulate(plateau, joueur):
    """Rollout aléatoire jusqu'à la fin"""
    p = plateau.copy()
    j = joueur
    while not est_terminé(p):
        coups = coups_valides(p, j)
        if coups:
            coup = coups[np.random.randint(len(coups))]
            p = jouer_coup(p, coup[0], coup[1], j)
        j = adversaire(j)
    return score_final(p)

def backpropagate(node, résultat, joueur_mcts):
    """Remonter — victoire si le joueur MCTS a gagné"""
    while node:
        node.visits += 1
        # Victoire du point de vue du joueur qui a joué CE nœud
        if résultat > 0 and node.joueur == adversaire(joueur_mcts):
            node.wins += 1
        elif résultat < 0 and node.joueur == joueur_mcts:
            node.wins += 1
        node = node.parent

def choisir_action_mcts(plateau, joueur, nb_simulations=200):
    racine = Node(plateau=plateau.copy(), joueur=joueur)

    for _ in range(nb_simulations):
        # 1. Select
        noeud = select(racine)

        # 2. Expand
        if not est_terminé(noeud.plateau):
            noeud = expand(noeud)

        # 3. Simulate
        résultat = simulate(noeud.plateau, noeud.joueur)

        # 4. Backpropagate
        backpropagate(noeud, résultat, joueur)

    if not racine.children:
        coups = coups_valides(plateau, joueur)
        return coups[0] if coups else None

    meilleur = max(racine.children, key=lambda n: n.visits)
    return meilleur.coup


# JOUER DES PARTIES — MCTS vs ALÉATOIRE


def jouer_partie(nb_simulations=200, afficher=False):
    plateau = créer_plateau()
    joueur = NOIR  # MCTS = NOIR, Aléatoire = BLANC

    while not est_terminé(plateau):
        coups = coups_valides(plateau, joueur)
        if coups:
            if joueur == NOIR:
                coup = choisir_action_mcts(plateau, joueur, nb_simulations)
            else:
                coup = coups[np.random.randint(len(coups))]
            if coup:
                plateau = jouer_coup(plateau, coup[0], coup[1], joueur)
        joueur = adversaire(joueur)

    if afficher:
        afficher_plateau(plateau)

    return score_final(plateau)


# TEST AVEC DIFFÉRENTES SIMULATIONS 

nb_parties = 10

for nb_sims in [50, 100, 200]:
    scores = []
    victoires = 0
    début = time.time()

    for i in range(nb_parties):
        résultat = jouer_partie(nb_simulations=nb_sims)
        scores.append(résultat)
        if résultat > 0:
            victoires += 1

    durée = time.time() - début
    print(f"\nMCTS {nb_sims} simulations — {nb_parties} parties")
    print(f"  Victoires     : {victoires}/{nb_parties} ({victoires/nb_parties*100:.0f}%)")
    print(f"  Temps total   : {durée:.1f}s")
    print(f"  Temps/partie  : {durée/nb_parties:.1f}s")
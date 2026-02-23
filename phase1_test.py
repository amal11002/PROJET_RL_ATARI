import gymnasium as gym
import ale_py

# Enregistrer les environnements Atari
gym.register_envs(ale_py)

# Lister quelques environnements disponibles pour vérifier l'installation
envs = [e for e in gym.envs.registry.keys() if "ALE" in e]
print(f"Nombre d'environnements Atari disponibles : {len(envs)}")
print("Exemples :", envs[:5])
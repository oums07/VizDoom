import random
import numpy as np
import torch
from torch import nn

from model import DQN


class DQNAgent: 
    def __init__(
        self,
        nb_actions=3,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        gamma=0.99,
        learning_rate=0.0001,
        target_update_frequency=1000
    ):
        # Paramètres de l'agent
        self.nb_actions = nb_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        # Réseau principal : celui qui apprend
        self.model = DQN(nb_actions)

        # Réseau cible : utilisé pour calculer les valeurs Q futures
        self.target_model = DQN(nb_actions)

        # Initialiser le réseau cible avec les mêmes poids
        self.target_model.load_state_dict(self.model.state_dict())

        # Désactiver l'apprentissage direct du réseau cible
        self.target_model.requires_grad_(False)
        self.target_model.eval()

        # Fonction de perte
        self.loss_fn = nn.SmoothL1Loss()

        # Optimiseur du réseau principal
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        # Compteur des mises à jour d'apprentissage
        self.training_steps = 0
        self.target_update_frequency = target_update_frequency

    def choisir_action(self, observation):
        # Exploration
        if random.random() < self.epsilon:
            return random.randrange(self.nb_actions)

        # Exploitation
        observation = torch.as_tensor(
            np.asarray(observation, dtype=np.float32)
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(observation)

        return q_values.argmax(dim=1).item()

    def diminuer_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def apprendre(self, memoire, batch_size=32):
        # Attendre suffisamment d'expériences
        if len(memoire) < batch_size:
            return None

        # Sélectionner des expériences au hasard
        batch = memoire.echantillonner(batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        # Convertir les données en tenseurs PyTorch
        states = torch.as_tensor(
            np.array(states), dtype=torch.float32
        )
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # En fin de partie, next_state peut être None
        next_states = [
            s if s is not None else np.zeros((1, 84, 84), dtype=np.float32)
            for s in next_states
        ]
        next_states = torch.as_tensor(
            np.array(next_states), dtype=torch.float32
        )

        # 1. Calculer les valeurs Q prédites
        q_values = self.model(states)
        q_pred = q_values.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # 2. Utiliser le réseau cible pendant l'apprentissage
        with torch.no_grad():
            q_next = self.target_model(next_states).max(dim=1).values
            q_target = rewards + self.gamma * q_next * (1 - dones)


        # 3. Calculer l'erreur
        loss = self.loss_fn(q_pred, q_target)

        # 4. Modifier les poids du réseau
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compter chaque mise à jour d'apprentissage
        self.training_steps += 1

        # Synchroniser le réseau cible périodiquement
        if self.training_steps % self.target_update_frequency == 0:
            self.mettre_a_jour_cible()

        return loss.item()

    def mettre_a_jour_cible(self):
        self.target_model.load_state_dict(
            self.model.state_dict()
        )

    
    def sauvegarder(self, chemin="dqn_model.pth"):
        sauvegarde = {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps
        }

        torch.save(sauvegarde, chemin)


    def charger(self, chemin="dqn_model.pth"):
        sauvegarde = torch.load(
            chemin,
            map_location="cpu",
            weights_only=True
        )

        self.model.load_state_dict(sauvegarde["model"])
        self.target_model.load_state_dict(sauvegarde["target_model"])
        self.optimizer.load_state_dict(sauvegarde["optimizer"])

        self.epsilon = sauvegarde["epsilon"]
        self.training_steps = sauvegarde["training_steps"]

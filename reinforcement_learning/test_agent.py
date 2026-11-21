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

        # Réseau principal utilisé pour choisir les actions et apprendre
        self.model = DQN(nb_actions)

        # Réseau cible utilisé pour calculer les valeurs Q futures
        self.target_model = DQN(nb_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.requires_grad_(False)
        self.target_model.eval()

        # Fonction de perte et optimiseur
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        # Compteur pour synchroniser le réseau cible
        self.training_steps = 0
        self.target_update_frequency = target_update_frequency

    def choisir_action(self, observation):
        # Exploration : choisir une action au hasard
        if random.random() < self.epsilon:
            return random.randrange(self.nb_actions)

        # Exploitation : préparer l'image pour le réseau
        observation = torch.as_tensor(
            np.asarray(observation, dtype=np.float32)
        ).unsqueeze(0)

        # Calculer les valeurs Q sans modifier les poids
        with torch.no_grad():
            q_values = self.model(observation)

        # Retourner l'action avec la plus grande valeur Q
        return q_values.argmax(dim=1).item()

    def diminuer_epsilon(self):
        # Réduire progressivement l'exploration
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def apprendre(self, memoire, batch_size=32):
        # Attendre d'avoir assez d'expériences
        if len(memoire) < batch_size:
            return None

        # Sélectionner un mini-batch dans la mémoire
        batch = memoire.echantillonner(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convertir les expériences en tenseurs PyTorch
        states = torch.as_tensor(
            np.array(states), dtype=torch.float32
        )
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Remplacer les observations None en fin de partie
        next_states = [
            s if s is not None else np.zeros((1, 84, 84), dtype=np.float32)
            for s in next_states
        ]
        next_states = torch.as_tensor(
            np.array(next_states), dtype=torch.float32
        )

        # Calculer les valeurs Q des actions réellement effectuées
        q_values = self.model(states)
        q_pred = q_values.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Calculer les valeurs Q cibles avec le réseau cible
        # Ne pas prendre en compte les récompenses futures si done=True
        with torch.no_grad():
            q_next = self.target_model(next_states).max(dim=1).values
            q_target = rewards + self.gamma * q_next * (1 - dones)

        # Mesurer l'erreur entre les prédictions et les cibles
        loss = self.loss_fn(q_pred, q_target)

        # Calculer les gradients et mettre à jour les poids du réseau
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compter les mises à jour et synchroniser périodiquement les réseaux
        self.training_steps += 1

        if self.training_steps % self.target_update_frequency == 0:
            self.mettre_a_jour_cible()

        return loss.item()

    def mettre_a_jour_cible(self):
        # Copier les poids du réseau principal vers le réseau cible
        self.target_model.load_state_dict(
            self.model.state_dict()
        )

    def sauvegarder(self, chemin="dqn_model.pth"):
        # Conserver les réseaux et les paramètres d'apprentissage
        sauvegarde = {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps
        }

        torch.save(sauvegarde, chemin)

    def charger(self, chemin="dqn_model.pth"):
        # Récupérer les paramètres sauvegardés
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
import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, nb_actions=3):
        super().__init__()

        # Extraire les caractéristiques de l'image
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculer une valeur Q pour chaque action
        self.action_values = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, nb_actions)
        )

    def forward(self, observation):
        features = self.features(observation)
        return self.action_values(features)


if __name__ == "__main__":
    model = DQN(nb_actions=3)

    # Un lot contenant une seule image en niveaux de gris
    observation = torch.zeros((1, 1, 84, 84))

    with torch.no_grad():
        q_values = model(observation)

    print("Forme de l'entrée :", observation.shape)
    print("Forme de la sortie :", q_values.shape)
    print("Valeurs Q :", q_values)
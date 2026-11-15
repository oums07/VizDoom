import csv
import random
import time
from datetime import datetime
from pathlib import Path

from environment import VizDoomEnv



results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(exist_ok=True)

csv_path = results_dir / "random_results.csv"


date_execution = datetime.now().isoformat(timespec="seconds")

# Ajouter les noms des colonnes seulement si nécessaire
ajouter_entete = (
    not csv_path.exists() or csv_path.stat().st_size == 0
)

env = VizDoomEnv(visible=True)

try:
    # "a" ajoute les résultats sans effacer les précédents
    with open(csv_path, "a", newline="", encoding="utf-8") as fichier:
        writer = csv.writer(fichier)

        if ajouter_entete:
            writer.writerow([
                "date_execution",
                "episode",
                "total_reward"
            ])

        for episode in range(5):
            observation = env.reset()
            print(
                "Forme :", observation.shape,
                "| Type :", observation.dtype,
                "| Min :", observation.min(),
                "| Max :", observation.max()
       )
            done = False
            total_reward = 0

            while not done:
                action = random.randrange(len(env.actions))

                observation, reward, done, infos = env.step(action)
                total_reward += reward

                # Ralentir pour observer le jeu
                time.sleep(0.02)

            writer.writerow([
                date_execution,
                episode + 1,
                total_reward
            ])
            fichier.flush()

            print(
                f"Partie {episode + 1} : "
                f"récompense totale = {total_reward}"
            )

    print(f"Résultats enregistrés dans : {csv_path}")

finally:
    env.close()
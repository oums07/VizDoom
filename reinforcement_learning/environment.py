from pathlib import Path
import vizdoom as vzd
import numpy as np
from PIL import Image

class VizDoomEnv:
    def __init__(self, visible=True):
        self.game = vzd.DoomGame()

        scenario = Path(vzd.scenarios_path) / "basic.cfg"
        self.game.load_config(str(scenario))
        # Limite de durée en tics de simulation
        self.game.set_episode_timeout(300)

        # Pénalité pour chaque tic écoulé
        self.game.set_living_reward(-1.0)
        self.game.set_window_visible(visible)
        self.game.set_mode(vzd.Mode.PLAYER)
        self.game.set_screen_resolution(
            vzd.ScreenResolution.RES_1280X720
        )

        self.game.set_available_buttons([
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.ATTACK
        ])

        self.actions = [
            [True, False, False],   # Action 0 : aller à gauche
            [False, True, False],   # Action 1 : aller à droite
            [False, False, True]    # Action 2 : tirer
        ]
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        self.game.init()

    def reset(self):
        self.game.new_episode()
        image = self.game.get_state().screen_buffer
        return self.preprocess(image)
    def preprocess(self, image):
        image = Image.fromarray(image)
        image = image.resize((84, 84), Image.Resampling.BILINEAR)

        observation = np.asarray(image, dtype=np.float32) / 255.0

        return observation[np.newaxis, :, :]

    def step(self, action):
        if not isinstance(action, (int, np.integer)):
            raise ValueError("L'action doit être un entier : 0, 1 ou 2.")

        if not 0 <= action < len(self.actions):
            raise ValueError("Action invalide : choisir 0, 1 ou 2.")

        if self.game.is_episode_finished():
            raise RuntimeError(
                "La partie est terminée. Appelle reset() avant de rejouer."
            )
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()

        if done:
            observation = None
        else:
            image = self.game.get_state().screen_buffer
            observation = self.preprocess(image)

        return observation, reward, done, {}

    def close(self):
        self.game.close()


if __name__ == "__main__":
    env = VizDoomEnv(visible=True)

    try:
        input("Jeu chargé. Appuie sur Entrée ici pour fermer.")
    finally:
        env.close()
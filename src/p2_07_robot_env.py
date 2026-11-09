
import gymnasium as gym
import numpy as np

from gymnasium import spaces


# ============================================
# PRACTICA 2 - SESION 7
# ENTORNO PERSONALIZADO DE ROBOT
# ============================================


class RobotLineEnv(gym.Env):

    metadata = {
        "render_modes": ["ansi"]
    }

    def __init__(self, meta=10, max_pasos=30):

        super().__init__()

        self.meta = meta
        self.max_pasos = max_pasos

        # Acciones:
        # 0 = izquierda
        # 1 = derecha
        self.action_space = spaces.Discrete(2)

        # Observacion: posicion del robot
        self.observation_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([float(meta)], dtype=np.float32),
            dtype=np.float32
        )

        self.posicion = 0
        self.pasos = 0

    def _observacion(self):
        return np.array(
            [float(self.posicion)],
            dtype=np.float32
        )

    def _informacion(self):
        return {
            "posicion": self.posicion,
            "meta": self.meta,
            "pasos": self.pasos
        }

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        self.posicion = 0
        self.pasos = 0

        return (
            self._observacion(),
            self._informacion()
        )

    def step(self, action):

        if not self.action_space.contains(action):
            raise ValueError(
                f"Accion no valida: {action}"
            )

        # Mover el robot
        if action == 0:
            self.posicion -= 1
        else:
            self.posicion += 1

        # Evitar salir de los limites
        self.posicion = int(
            np.clip(self.posicion, 0, self.meta)
        )

        self.pasos += 1

        # Condicion de exito
        terminado = self.posicion == self.meta

        # Limite de pasos
        truncado = (
            self.pasos >= self.max_pasos
            and not terminado
        )

        # Sistema de recompensas
        if terminado:
            recompensa = 100.0
        else:
            recompensa = -1.0

        return (
            self._observacion(),
            recompensa,
            terminado,
            truncado,
            self._informacion()
        )

    def render(self):

        linea = ["."] * (self.meta + 1)

        linea[self.posicion] = "R"

        if self.posicion != self.meta:
            linea[self.meta] = "M"

        return " ".join(linea)

    def close(self):
        pass

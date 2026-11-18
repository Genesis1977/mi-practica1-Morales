
"""
PRACTICA 2 - SESION 7
Entorno personalizado de aprendizaje por refuerzo.

El agente debe desplazarse en una linea
hasta alcanzar una meta.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RobotLineEnv(gym.Env):
    """Entorno de navegacion lineal para un agente."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, longitud=10, max_pasos=30):
        super().__init__()

        self.longitud = longitud
        self.max_pasos = max_pasos

        # Acciones:
        # 0 = izquierda
        # 1 = derecha
        self.action_space = spaces.Discrete(2)

        # Observacion: posicion actual del agente
        self.observation_space = spaces.Box(
            low=0,
            high=longitud,
            shape=(1,),
            dtype=np.float32
        )

        self.posicion = 0
        self.pasos = 0

    def _observacion(self):
        """Devuelve la posicion actual."""
        return np.array(
            [self.posicion],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reinicia el entorno."""
        super().reset(seed=seed)

        self.posicion = 0
        self.pasos = 0

        return self._observacion(), {}

    def step(self, action):
        """Ejecuta una accion y calcula recompensa."""

        if not self.action_space.contains(action):
            raise ValueError("Accion no valida")

        self.pasos += 1

        if action == 0:
            self.posicion = max(
                0, self.posicion - 1
            )
        else:
            self.posicion = min(
                self.longitud,
                self.posicion + 1
            )

        terminated = self.posicion == self.longitud
        truncated = (
            self.pasos >= self.max_pasos
            and not terminated
        )

        # Penalizacion por cada paso
        recompensa = -1.0

        # Recompensa al alcanzar la meta
        if terminated:
            recompensa = 100.0

        info = {
            "posicion": self.posicion,
            "meta_alcanzada": terminated
        }

        return (
            self._observacion(),
            recompensa,
            terminated,
            truncated,
            info
        )

    def render(self):
        """Muestra la posicion del agente."""
        linea = ["."] * (self.longitud + 1)

        linea[self.posicion] = "R"

        if self.posicion != self.longitud:
            linea[self.longitud] = "M"

        print(" ".join(linea))


def main():
    """Prueba el entorno con acciones aleatorias."""

    env = RobotLineEnv()

    recompensas = []

    for episodio in range(1, 6):

        estado, info = env.reset(
            seed=42 + episodio
        )

        recompensa_total = 0
        terminado = False

        print(f"\n=== EPISODIO {episodio} ===")

        while not terminado:

            accion = env.action_space.sample()

            (
                estado,
                recompensa,
                terminated,
                truncated,
                info
            ) = env.step(accion)

            terminado = terminated or truncated
            recompensa_total += recompensa

            print(
                f"Accion: {accion} | "
                f"Posicion: {estado[0]:.0f} | "
                f"Recompensa: {recompensa}"
            )

            env.render()

        recompensas.append(recompensa_total)

        print(
            "Meta alcanzada:",
            info["meta_alcanzada"]
        )

        print(
            "Recompensa acumulada:",
            recompensa_total
        )

    print("\n=== RESULTADOS FINALES ===")
    print(
        "Recompensa promedio:",
        np.mean(recompensas)
    )
    print(
        "Recompensa maxima:",
        np.max(recompensas)
    )
    print(
        "Recompensa minima:",
        np.min(recompensas)
    )

    env.close()


if __name__ == "__main__":
    main()

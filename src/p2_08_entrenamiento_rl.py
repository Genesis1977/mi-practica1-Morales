
"""
PRACTICA 2 - SESION 8
Entrenamiento de un agente PPO en CartPole-v1.

Entregables:
- models/ppo_cartpole.zip
- reports/figures/p2_08_recompensas.png
- reports/p2_08_entrenamiento.json
"""

from pathlib import Path
import json

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


BASE = Path(__file__).resolve().parent.parent

MODELOS = BASE / "models"
REPORTES = BASE / "reports"
FIGURAS = REPORTES / "figures"

MODELOS.mkdir(parents=True, exist_ok=True)
REPORTES.mkdir(parents=True, exist_ok=True)
FIGURAS.mkdir(parents=True, exist_ok=True)

PASOS_ENTRENAMIENTO = 20000
SEMILLA = 42


class RegistroRecompensas(BaseCallback):
    """Registra recompensas de episodios completados."""

    def __init__(self):
        super().__init__()
        self.recompensas = []
        self.duraciones = []
        self.pasos = []

    def _on_step(self):
        """Captura los resultados reportados por Monitor."""

        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                episodio = info["episode"]

                self.recompensas.append(
                    float(episodio["r"])
                )

                self.duraciones.append(
                    int(episodio["l"])
                )

                self.pasos.append(
                    int(self.num_timesteps)
                )

        return True


def crear_entorno():
    """Crea CartPole con monitoreo de episodios."""

    entorno = gym.make("CartPole-v1")

    entorno.reset(seed=SEMILLA)
    entorno.action_space.seed(SEMILLA)

    return Monitor(entorno)


def entrenar_agente():
    """Entrena PPO y guarda el modelo."""

    print("\n=== SESION 8: ENTRENAMIENTO PPO ===")

    entorno = crear_entorno()

    callback = RegistroRecompensas()

    modelo = PPO(
        policy="MlpPolicy",
        env=entorno,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        seed=SEMILLA,
        verbose=1,
        device="cpu"
    )

    print("\nIniciando entrenamiento...")

    modelo.learn(
        total_timesteps=PASOS_ENTRENAMIENTO,
        callback=callback
    )

    ruta_modelo = MODELOS / "ppo_cartpole"

    modelo.save(str(ruta_modelo))

    print("\nModelo guardado:")
    print(ruta_modelo.with_suffix(".zip"))

    entorno.close()

    return callback


def graficar_recompensas(callback):
    """Genera grafico de recompensas acumuladas."""

    recompensas = callback.recompensas

    if not recompensas:
        raise RuntimeError(
            "No se registraron episodios."
        )

    ventana = min(20, len(recompensas))

    promedios = []

    for i in range(len(recompensas)):
        inicio = max(0, i - ventana + 1)

        promedio = np.mean(
            recompensas[inicio:i + 1]
        )

        promedios.append(promedio)

    plt.figure(figsize=(11, 6))

    plt.plot(
        callback.pasos,
        recompensas,
        alpha=0.35,
        label="Recompensa por episodio"
    )

    plt.plot(
        callback.pasos,
        promedios,
        linewidth=2,
        label=f"Media movil ({ventana} episodios)"
    )

    plt.title(
        "Practica 2 - Sesion 8: Entrenamiento PPO"
    )

    plt.xlabel("Pasos de entrenamiento")
    plt.ylabel("Recompensa acumulada")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    ruta = FIGURAS / "p2_08_recompensas.png"

    plt.savefig(ruta, dpi=300)
    plt.close()

    print("\nGrafico guardado:")
    print(ruta)


def guardar_resultados(callback):
    """Guarda estadisticas del entrenamiento."""

    recompensas = callback.recompensas

    resultados = {
        "sesion": 8,
        "algoritmo": "PPO",
        "entorno": "CartPole-v1",
        "pasos_solicitados": PASOS_ENTRENAMIENTO,
        "pasos_reales": callback.num_timesteps,
        "episodios_completados": len(recompensas),
        "recompensa_promedio": float(
            np.mean(recompensas)
        ),
        "recompensa_maxima": float(
            np.max(recompensas)
        ),
        "recompensa_minima": float(
            np.min(recompensas)
        ),
        "promedio_ultimos_10": float(
            np.mean(recompensas[-10:])
        ),
        "semilla": SEMILLA
    }

    ruta = REPORTES / "p2_08_entrenamiento.json"

    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(
            resultados,
            archivo,
            indent=4,
            ensure_ascii=False
        )

    print("\n=== RESULTADOS ===")

    for clave, valor in resultados.items():
        print(f"{clave}: {valor}")

    print("\nReporte guardado:")
    print(ruta)


def main():
    """Ejecuta el entrenamiento y genera evidencias."""

    callback = entrenar_agente()

    graficar_recompensas(callback)

    guardar_resultados(callback)

    print("\nSESION 8 COMPLETADA")


if __name__ == "__main__":
    main()


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from p2_07_robot_env import RobotLineEnv


# ============================================
# PRACTICA 2 - SESION 8
# ENTRENAMIENTO DE AGENTE PPO
# ============================================

BASE = Path(__file__).resolve().parent.parent

MODELOS = BASE / "models"
DATOS = BASE / "data" / "processed"
FIGURAS = BASE / "reports" / "figures"

MODELOS.mkdir(parents=True, exist_ok=True)
DATOS.mkdir(parents=True, exist_ok=True)
FIGURAS.mkdir(parents=True, exist_ok=True)

TOTAL_TIMESTEPS = 10000
SEMILLA = 42


# --------------------------------------------
# CALLBACK PARA REGISTRAR EPISODIOS
# --------------------------------------------

class RegistroEpisodios(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

        self.resultados = []

    def _on_step(self):

        infos = self.locals.get("infos", [])

        for info in infos:

            if "episode" in info:

                episodio = info["episode"]

                self.resultados.append({
                    "Episodio": len(self.resultados) + 1,
                    "Paso global": self.num_timesteps,
                    "Recompensa": float(episodio["r"]),
                    "Duracion": int(episodio["l"])
                })

        return True


def main():

    print("\n=== SESION 8: ENTRENAMIENTO PPO ===")

    # ----------------------------------------
    # 1. COMPROBAR ENTORNO
    # ----------------------------------------

    entorno_prueba = RobotLineEnv(
        meta=10,
        max_pasos=30
    )

    check_env(entorno_prueba, warn=True)
    entorno_prueba.close()

    print("Entorno personalizado verificado.")

    # ----------------------------------------
    # 2. CREAR ENTORNO DE ENTRENAMIENTO
    # ----------------------------------------

    entorno = Monitor(
        RobotLineEnv(
            meta=10,
            max_pasos=30
        )
    )

    # ----------------------------------------
    # 3. CONFIGURAR PPO
    # ----------------------------------------

    modelo = PPO(
        policy="MlpPolicy",
        env=entorno,
        learning_rate=0.0003,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        seed=SEMILLA,
        verbose=1,
        device="cpu"
    )

    callback = RegistroEpisodios()

    # ----------------------------------------
    # 4. ENTRENAR EL AGENTE
    # ----------------------------------------

    print("\nIniciando entrenamiento...")
    print("Pasos solicitados:", TOTAL_TIMESTEPS)

    modelo.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=False
    )

    print("\nEntrenamiento finalizado.")
    print("Pasos realizados:", modelo.num_timesteps)

    # ----------------------------------------
    # 5. GUARDAR MODELO
    # ----------------------------------------

    ruta_modelo = MODELOS / "p2_ppo_robot"

    modelo.save(str(ruta_modelo))

    print("\nModelo guardado:")
    print(str(ruta_modelo) + ".zip")

    # ----------------------------------------
    # 6. GUARDAR RESULTADOS
    # ----------------------------------------

    tabla = pd.DataFrame(callback.resultados)

    if tabla.empty:
        entorno.close()
        raise RuntimeError(
            "No se registraron episodios."
        )

    ruta_csv = DATOS / "p2_ppo_entrenamiento.csv"

    tabla.to_csv(
        ruta_csv,
        index=False
    )

    # ----------------------------------------
    # 7. CALCULAR PROMEDIO MOVIL
    # ----------------------------------------

    tabla["Promedio movil"] = (
        tabla["Recompensa"]
        .rolling(window=20, min_periods=1)
        .mean()
    )

    # ----------------------------------------
    # 8. GENERAR GRAFICA
    # ----------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        tabla["Episodio"],
        tabla["Recompensa"],
        alpha=0.35,
        label="Recompensa por episodio"
    )

    plt.plot(
        tabla["Episodio"],
        tabla["Promedio movil"],
        linewidth=2,
        label="Promedio movil (20 episodios)"
    )

    plt.xlabel("Episodio")
    plt.ylabel("Recompensa acumulada")

    plt.title(
        "Entrenamiento PPO - RobotLineEnv"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    ruta_grafica = (
        FIGURAS / "p2_ppo_entrenamiento.png"
    )

    plt.savefig(
        ruta_grafica,
        dpi=300
    )

    plt.close()

    # ----------------------------------------
    # 9. MOSTRAR RESUMEN
    # ----------------------------------------

    print("\n=== RESUMEN DEL ENTRENAMIENTO ===")

    print("Episodios registrados:", len(tabla))

    print(
        "Recompensa promedio general:",
        round(tabla["Recompensa"].mean(), 2)
    )

    print(
        "Promedio primeros 20 episodios:",
        round(tabla["Recompensa"].head(20).mean(), 2)
    )

    print(
        "Promedio ultimos 20 episodios:",
        round(tabla["Recompensa"].tail(20).mean(), 2)
    )

    print("\n=== ARCHIVOS GENERADOS ===")
    print(str(ruta_modelo) + ".zip")
    print(ruta_csv)
    print(ruta_grafica)

    entorno.close()

    print("\nSesion 8 completada correctamente.")


if __name__ == "__main__":
    main()

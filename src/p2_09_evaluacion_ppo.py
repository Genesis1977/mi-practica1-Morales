
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from p2_07_robot_env import RobotLineEnv


# ============================================
# PRACTICA 2 - SESION 9
# EVALUACION PPO VS ALEATORIO VS REGLAS
# ============================================

BASE = Path(__file__).resolve().parent.parent

MODELO = BASE / "models" / "p2_ppo_robot.zip"
DATOS = BASE / "data" / "processed"
FIGURAS = BASE / "reports" / "figures"

DATOS.mkdir(parents=True, exist_ok=True)
FIGURAS.mkdir(parents=True, exist_ok=True)

N_EPISODIOS = 30
SEMILLA = 100


def ejecutar_episodio(entorno, politica, modelo, numero):

    observacion, info = entorno.reset(
        seed=SEMILLA + numero
    )

    entorno.action_space.seed(
        SEMILLA + numero
    )

    recompensa_total = 0.0
    terminado = False
    truncado = False
    pasos = 0

    while not (terminado or truncado):

        if politica == "Aleatorio":
            accion = entorno.action_space.sample()

        elif politica == "Reglas":
            accion = 1

        elif politica == "PPO":
            accion, _ = modelo.predict(
                observacion,
                deterministic=True
            )

        else:
            raise ValueError(
                f"Politica desconocida: {politica}"
            )

        (
            observacion,
            recompensa,
            terminado,
            truncado,
            info
        ) = entorno.step(accion)

        recompensa_total += recompensa
        pasos += 1

    return {
        "Politica": politica,
        "Episodio": numero,
        "Recompensa": recompensa_total,
        "Pasos": pasos,
        "Exito": bool(terminado),
        "Truncado": bool(truncado)
    }


def main():

    print("\n=== SESION 9: EVALUACION DE AGENTES ===")

    if not MODELO.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo PPO: {MODELO}"
        )

    # Cargar el modelo entrenado
    modelo = PPO.load(
        str(MODELO),
        device="cpu"
    )

    politicas = [
        "Aleatorio",
        "Reglas",
        "PPO"
    ]

    resultados = []

    # Evaluar cada politica
    for politica in politicas:

        entorno = RobotLineEnv(
            meta=10,
            max_pasos=30
        )

        print(f"\n=== POLITICA: {politica} ===")

        for episodio in range(1, N_EPISODIOS + 1):

            resultado = ejecutar_episodio(
                entorno,
                politica,
                modelo,
                episodio
            )

            resultados.append(resultado)

            print(
                f"Episodio {episodio}: "
                f"recompensa={resultado['Recompensa']:.2f}, "
                f"pasos={resultado['Pasos']}, "
                f"exito={resultado['Exito']}"
            )

        entorno.close()

    # ----------------------------------------
    # CREAR TABLAS
    # ----------------------------------------

    tabla = pd.DataFrame(resultados)

    resumen = (
        tabla.groupby("Politica")
        .agg(
            recompensa_promedio=("Recompensa", "mean"),
            recompensa_std=("Recompensa", "std"),
            pasos_promedio=("Pasos", "mean"),
            tasa_exito=("Exito", "mean")
        )
        .reindex(politicas)
        .reset_index()
    )

    resumen["tasa_exito"] *= 100

    print("\n=== COMPARACION FINAL ===")
    print(resumen.to_string(index=False))

    # ----------------------------------------
    # GUARDAR CSV
    # ----------------------------------------

    ruta_detalle = (
        DATOS / "p2_09_episodios.csv"
    )

    ruta_resumen = (
        DATOS / "p2_09_comparacion.csv"
    )

    tabla.to_csv(
        ruta_detalle,
        index=False
    )

    resumen.to_csv(
        ruta_resumen,
        index=False
    )

    # ----------------------------------------
    # GRAFICA DE RECOMPENSAS
    # ----------------------------------------

    plt.figure(figsize=(9, 5))

    plt.bar(
        resumen["Politica"],
        resumen["recompensa_promedio"]
    )

    plt.title(
        "Comparacion de recompensa promedio"
    )
    plt.xlabel("Politica")
    plt.ylabel("Recompensa promedio")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    ruta_recompensas = (
        FIGURAS / "p2_09_recompensas.png"
    )

    plt.savefig(
        ruta_recompensas,
        dpi=300
    )

    plt.close()

    # ----------------------------------------
    # GRAFICA DE TASA DE EXITO
    # ----------------------------------------

    plt.figure(figsize=(9, 5))

    plt.bar(
        resumen["Politica"],
        resumen["tasa_exito"]
    )

    plt.ylim(0, 105)
    plt.title("Tasa de exito por politica")
    plt.xlabel("Politica")
    plt.ylabel("Exito (%)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    ruta_exito = (
        FIGURAS / "p2_09_tasa_exito.png"
    )

    plt.savefig(
        ruta_exito,
        dpi=300
    )

    plt.close()

    print("\n=== ARCHIVOS GENERADOS ===")
    print(ruta_detalle)
    print(ruta_resumen)
    print(ruta_recompensas)
    print(ruta_exito)

    print("\nSesion 9 completada correctamente.")


if __name__ == "__main__":
    main()

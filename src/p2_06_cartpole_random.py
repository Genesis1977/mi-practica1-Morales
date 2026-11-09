
from pathlib import Path

import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt


# ============================================
# PRACTICA 2 - SESION 6
# AGENTE ALEATORIO EN CARTPOLE
# ============================================

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / "data" / "processed"
FIGURAS = BASE / "reports" / "figures"

DATOS.mkdir(parents=True, exist_ok=True)
FIGURAS.mkdir(parents=True, exist_ok=True)

N_EPISODIOS = 5
SEMILLA = 42


def main():

    print("\n=== SESION 6: CARTPOLE ===")

    # Crear entorno sin ventana grafica
    entorno = gym.make("CartPole-v1")

    print("Entorno:", entorno.spec.id)
    print("Espacio de observacion:", entorno.observation_space)
    print("Espacio de acciones:", entorno.action_space)

    resultados = []

    # Ejecutar cinco episodios
    for episodio in range(1, N_EPISODIOS + 1):

        observacion, info = entorno.reset(
            seed=SEMILLA + episodio
        )

        entorno.action_space.seed(
            SEMILLA + episodio
        )

        recompensa_total = 0.0
        pasos = 0
        terminado = False
        truncado = False

        while not (terminado or truncado):

            # Elegir una accion aleatoria
            accion = entorno.action_space.sample()

            # Ejecutar la accion
            observacion, recompensa, terminado, truncado, info = (
                entorno.step(accion)
            )

            recompensa_total += recompensa
            pasos += 1

        resultados.append({
            "Episodio": episodio,
            "Pasos": pasos,
            "Recompensa total": recompensa_total,
            "Terminado": terminado,
            "Truncado": truncado
        })

        print(
            f"Episodio {episodio}: "
            f"pasos={pasos}, "
            f"recompensa={recompensa_total:.2f}"
        )

    entorno.close()

    # Crear tabla
    tabla = pd.DataFrame(resultados)

    promedio = tabla["Recompensa total"].mean()
    minimo = tabla["Recompensa total"].min()
    maximo = tabla["Recompensa total"].max()

    print("\n=== RESUMEN ===")
    print(tabla.to_string(index=False))
    print(f"Recompensa promedio: {promedio:.2f}")
    print(f"Recompensa minima: {minimo:.2f}")
    print(f"Recompensa maxima: {maximo:.2f}")

    # Guardar CSV
    ruta_csv = DATOS / "p2_cartpole_random.csv"
    tabla.to_csv(ruta_csv, index=False)

    # Generar grafica
    plt.figure(figsize=(9, 5))

    plt.bar(
        tabla["Episodio"],
        tabla["Recompensa total"]
    )

    plt.axhline(
        promedio,
        linestyle="--",
        label=f"Promedio: {promedio:.2f}"
    )

    plt.xlabel("Episodio")
    plt.ylabel("Recompensa acumulada")
    plt.title("CartPole-v1 - Agente aleatorio")
    plt.xticks(tabla["Episodio"])
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    ruta_grafico = FIGURAS / "p2_cartpole_random.png"

    plt.savefig(ruta_grafico, dpi=300)
    plt.close()

    print("\n=== ARCHIVOS GENERADOS ===")
    print(ruta_csv)
    print(ruta_grafico)

    print("\nSesion 6 completada correctamente.")


if __name__ == "__main__":
    main()

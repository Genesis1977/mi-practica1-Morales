
from pathlib import Path

import pandas as pd
from stable_baselines3.common.env_checker import check_env

from p2_07_robot_env import RobotLineEnv


# ============================================
# PRACTICA 2 - SESION 7
# PRUEBA DEL ENTORNO PERSONALIZADO
# ============================================

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / "data" / "processed"

DATOS.mkdir(parents=True, exist_ok=True)

N_EPISODIOS = 5
SEMILLA = 42


def ejecutar_episodio(entorno, numero, politica):

    observacion, info = entorno.reset(
        seed=SEMILLA + numero
    )

    entorno.action_space.seed(
        SEMILLA + numero
    )

    recompensa_total = 0.0
    terminado = False
    truncado = False

    while not (terminado or truncado):

        if politica == "aleatoria":
            accion = entorno.action_space.sample()
        else:
            # Politica sencilla: avanzar a la meta
            accion = 1

        (
            observacion,
            recompensa,
            terminado,
            truncado,
            info
        ) = entorno.step(accion)

        recompensa_total += recompensa

    return {
        "Politica": politica,
        "Episodio": numero,
        "Pasos": info["pasos"],
        "Posicion final": info["posicion"],
        "Recompensa total": recompensa_total,
        "Meta alcanzada": terminado
    }


def main():

    print("\n=== SESION 7: ENTORNO PERSONALIZADO ===")

    # Verificar compatibilidad con Gymnasium
    entorno_verificacion = RobotLineEnv()

    check_env(
        entorno_verificacion,
        warn=True
    )

    entorno_verificacion.close()

    print("Verificacion del entorno: correcta.")

    entorno = RobotLineEnv(
        meta=10,
        max_pasos=30
    )

    resultados = []

    # Comparar dos politicas
    for politica in ["aleatoria", "reglas"]:

        print(f"\n=== POLITICA: {politica} ===")

        for episodio in range(1, N_EPISODIOS + 1):

            resultado = ejecutar_episodio(
                entorno,
                episodio,
                politica
            )

            resultados.append(resultado)

            print(
                f"Episodio {episodio}: "
                f"pasos={resultado['Pasos']}, "
                f"recompensa={resultado['Recompensa total']:.2f}, "
                f"meta={resultado['Meta alcanzada']}"
            )

    entorno.close()

    tabla = pd.DataFrame(resultados)

    print("\n=== RESULTADOS ===")
    print(tabla.to_string(index=False))

    print("\n=== PROMEDIOS POR POLITICA ===")

    resumen = tabla.groupby("Politica").agg(
        recompensa_promedio=(
            "Recompensa total", "mean"
        ),
        pasos_promedio=("Pasos", "mean"),
        tasa_exito=("Meta alcanzada", "mean")
    )

    print(resumen.to_string())

    ruta_csv = DATOS / "p2_robot_env_resultados.csv"

    tabla.to_csv(
        ruta_csv,
        index=False
    )

    print("\nArchivo generado:", ruta_csv)
    print("\nSesion 7 completada correctamente.")


if __name__ == "__main__":
    main()

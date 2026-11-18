
"""
PRACTICA 2 - SESION 10

Comparacion entre:
1. Agente PPO entrenado.
2. Controlador clasico basado en reglas.

Entregables:
- reports/p2_10_comparacion.csv
- reports/p2_10_resultados.json
- reports/p2_10_conclusiones.md
- reports/figures/p2_10_comparacion.png
"""

from pathlib import Path
import json

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stable_baselines3 import PPO


BASE = Path(__file__).resolve().parent.parent

MODELO = BASE / "models" / "ppo_cartpole.zip"
REPORTES = BASE / "reports"
FIGURAS = REPORTES / "figures"

REPORTES.mkdir(parents=True, exist_ok=True)
FIGURAS.mkdir(parents=True, exist_ok=True)

EPISODIOS = 10
SEMILLA = 42


def controlador_clasico(estado):
    """
    Selecciona una accion mediante reglas.

    Estado de CartPole:
    [posicion, velocidad, angulo, velocidad_angular]

    Accion:
    0 = izquierda
    1 = derecha
    """

    posicion, velocidad, angulo, velocidad_angular = estado

    # Regla heuristica: combinar inclinacion,
    # movimiento angular y desplazamiento del carrito.
    control = (
        1.0 * angulo
        + 0.5 * velocidad_angular
        + 0.05 * posicion
        + 0.1 * velocidad
    )

    if control > 0:
        return 1

    return 0


def evaluar_controlador(tipo, modelo=None):
    """Evalua PPO o controlador clasico en 10 episodios."""

    entorno = gym.make("CartPole-v1")

    resultados = []

    for episodio in range(EPISODIOS):

        semilla = SEMILLA + episodio

        estado, info = entorno.reset(seed=semilla)

        terminado = False
        recompensa_total = 0.0
        pasos = 0

        while not terminado:

            if tipo == "PPO":
                accion, _ = modelo.predict(
                    estado,
                    deterministic=True
                )
                accion = int(accion)

            elif tipo == "Clasico":
                accion = controlador_clasico(estado)

            else:
                raise ValueError(
                    f"Tipo de controlador desconocido: {tipo}"
                )

            estado, recompensa, terminated, truncated, info = (
                entorno.step(accion)
            )

            recompensa_total += float(recompensa)
            pasos += 1

            terminado = terminated or truncated

        resultados.append({
            "agente": tipo,
            "episodio": episodio + 1,
            "recompensa": recompensa_total,
            "pasos": pasos
        })

        print(
            f"{tipo} | Episodio {episodio + 1} | "
            f"Recompensa: {recompensa_total:.0f} | "
            f"Pasos: {pasos}"
        )

    entorno.close()

    return resultados


def estadisticas(resultados):
    """Calcula las metricas de un controlador."""

    recompensas = [
        fila["recompensa"] for fila in resultados
    ]

    return {
        "promedio": float(np.mean(recompensas)),
        "maximo": float(np.max(recompensas)),
        "minimo": float(np.min(recompensas)),
        "desviacion_estandar": float(
            np.std(recompensas)
        ),
        "episodios": len(recompensas)
    }


def graficar_comparacion(ppo, clasico):
    """Genera grafico de recompensa por episodio."""

    episodios = np.arange(1, EPISODIOS + 1)

    recompensa_ppo = [
        fila["recompensa"] for fila in ppo
    ]

    recompensa_clasico = [
        fila["recompensa"] for fila in clasico
    ]

    plt.figure(figsize=(11, 6))

    plt.plot(
        episodios,
        recompensa_ppo,
        marker="o",
        linewidth=2,
        label="PPO"
    )

    plt.plot(
        episodios,
        recompensa_clasico,
        marker="s",
        linewidth=2,
        label="Controlador clasico"
    )

    plt.title(
        "Sesion 10 - PPO vs Controlador Clasico"
    )

    plt.xlabel("Episodio")
    plt.ylabel("Recompensa acumulada")

    plt.xticks(episodios)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    ruta = FIGURAS / "p2_10_comparacion.png"

    plt.savefig(ruta, dpi=300)
    plt.close()

    print("\nGrafico guardado:", ruta)


def guardar_resultados(ppo, clasico):
    """Exporta tabla, metricas y conclusiones."""

    tabla = pd.DataFrame(ppo + clasico)

    ruta_csv = REPORTES / "p2_10_comparacion.csv"

    tabla.to_csv(
        ruta_csv,
        index=False
    )

    metricas_ppo = estadisticas(ppo)
    metricas_clasico = estadisticas(clasico)

    diferencia = (
        metricas_ppo["promedio"]
        - metricas_clasico["promedio"]
    )

    resultados = {
        "sesion": 10,
        "entorno": "CartPole-v1",
        "episodios_por_agente": EPISODIOS,
        "PPO": metricas_ppo,
        "Clasico": metricas_clasico,
        "diferencia_promedio": float(diferencia)
    }

    ruta_json = REPORTES / "p2_10_resultados.json"

    with open(
        ruta_json,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            resultados,
            archivo,
            indent=4,
            ensure_ascii=False
        )

    if diferencia > 0:
        interpretacion = (
            "PPO obtuvo una recompensa promedio mayor "
            "que el controlador basado en reglas."
        )
    elif diferencia < 0:
        interpretacion = (
            "El controlador basado en reglas obtuvo una "
            "recompensa promedio mayor que PPO."
        )
    else:
        interpretacion = (
            "PPO y el controlador clasico obtuvieron "
            "la misma recompensa promedio."
        )

    conclusion = f"""# Practica 2 - Sesion 10

## Objetivo

Comparar un agente entrenado con PPO y un
controlador clasico basado en reglas
en el entorno CartPole-v1.

## Metodologia

Se evaluaron {EPISODIOS} episodios por agente,
utilizando las mismas semillas iniciales.

El agente PPO emplea una politica aprendida.
El controlador clasico usa reglas sobre
el angulo, la velocidad angular, la posicion
y la velocidad del carrito.

## Resultados

| Metrica | PPO | Clasico |
|---|---:|---:|
| Promedio | {metricas_ppo['promedio']:.2f} | {metricas_clasico['promedio']:.2f} |
| Maximo | {metricas_ppo['maximo']:.2f} | {metricas_clasico['maximo']:.2f} |
| Minimo | {metricas_ppo['minimo']:.2f} | {metricas_clasico['minimo']:.2f} |
| Desviacion estandar | {metricas_ppo['desviacion_estandar']:.2f} | {metricas_clasico['desviacion_estandar']:.2f} |

Diferencia promedio PPO - Clasico: {diferencia:.2f}

## Interpretacion

{interpretacion}

## Ventajas y desventajas

### PPO

Ventajas:
- Aprende mediante experiencia y recompensas.
- No requiere programar manualmente cada decision.
- Puede adaptarse a problemas complejos mediante entrenamiento.

Desventajas:
- Requiere tiempo y recursos de entrenamiento.
- Su politica es menos interpretable.
- Su rendimiento depende del entrenamiento y la configuracion.

### Controlador clasico

Ventajas:
- No necesita entrenamiento.
- Sus decisiones son explicitas e interpretables.
- Tiene un costo computacional bajo.

Desventajas:
- Depende de reglas disenadas manualmente.
- Puede requerir ajustes para otras condiciones.
- No aprende automaticamente de la experiencia.

## Conclusion

La comparacion permite determinar cual enfoque
obtuvo mayor recompensa en los episodios evaluados.
Los resultados corresponden al entorno simulado
y no garantizan el mismo rendimiento en otros escenarios.
"""

    ruta_md = REPORTES / "p2_10_conclusiones.md"

    ruta_md.write_text(
        conclusion,
        encoding="utf-8"
    )

    print("\n=== COMPARACION FINAL ===")

    print("\nPPO:", metricas_ppo)
    print("CLASICO:", metricas_clasico)

    print(
        "\nDiferencia promedio:",
        round(diferencia, 2)
    )

    print("\nArchivos generados:")
    print(ruta_csv)
    print(ruta_json)
    print(ruta_md)


def main():
    """Ejecuta la comparacion completa."""

    print("\n=== PRACTICA 2 - SESION 10 ===")

    if not MODELO.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo: {MODELO}"
        )

    modelo = PPO.load(
        str(MODELO),
        device="cpu"
    )

    print("\nEvaluando PPO...")

    resultados_ppo = evaluar_controlador(
        "PPO",
        modelo=modelo
    )

    print("\nEvaluando controlador clasico...")

    resultados_clasico = evaluar_controlador(
        "Clasico"
    )

    graficar_comparacion(
        resultados_ppo,
        resultados_clasico
    )

    guardar_resultados(
        resultados_ppo,
        resultados_clasico
    )

    print("\nSESION 10 COMPLETADA")


if __name__ == "__main__":
    main()

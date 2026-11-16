
from pathlib import Path
from datetime import datetime
import json
import pandas as pd


# ============================================
# PRACTICA 2 - SESION 12
# PREPARACION DE ENTREGA FINAL
# ============================================

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / "data" / "processed"
REPORTES = BASE / "reports"

REPORTES.mkdir(parents=True, exist_ok=True)


def leer_json(ruta):

    if not ruta.exists():
        raise FileNotFoundError(
            f"Archivo requerido no encontrado: {ruta}"
        )

    return json.loads(
        ruta.read_text(encoding="utf-8")
    )


def leer_csv(ruta):

    if not ruta.exists():
        raise FileNotFoundError(
            f"Archivo requerido no encontrado: {ruta}"
        )

    return pd.read_csv(ruta)


def tabla_markdown(df):

    columnas = list(df.columns)

    encabezado = "| " + " | ".join(columnas) + " |"
    separador = "| " + " | ".join(
        ["---"] * len(columnas)
    ) + " |"

    filas = []

    for _, fila in df.iterrows():

        valores = [
            str(valor)
            for valor in fila.tolist()
        ]

        filas.append(
            "| " + " | ".join(valores) + " |"
        )

    return "\n".join(
        [encabezado, separador] + filas
    )


def main():

    print("\n=== SESION 12: ENTREGA FINAL ===")

    auditoria = leer_json(
        REPORTES / "p2_11_auditoria.json"
    )

    comparacion = leer_csv(
        DATOS / "p2_09_comparacion.csv"
    )

    columnas = [
        "Politica",
        "recompensa_promedio",
        "pasos_promedio",
        "tasa_exito"
    ]

    comparacion = comparacion[columnas].copy()

    comparacion[
        "recompensa_promedio"
    ] = comparacion[
        "recompensa_promedio"
    ].round(2)

    comparacion[
        "pasos_promedio"
    ] = comparacion[
        "pasos_promedio"
    ].round(2)

    comparacion[
        "tasa_exito"
    ] = comparacion[
        "tasa_exito"
    ].round(2)

    resultados = comparacion.set_index("Politica")

    ppo = resultados.loc["PPO"]
    aleatorio = resultados.loc["Aleatorio"]
    reglas = resultados.loc["Reglas"]

    recompensa_ppo = ppo["recompensa_promedio"]
    recompensa_aleatorio = aleatorio["recompensa_promedio"]
    recompensa_reglas = reglas["recompensa_promedio"]

    if recompensa_ppo == recompensa_reglas:
        conclusion = (
            "El agente PPO igualo la recompensa promedio "
            "de la politica basada en reglas."
        )

    elif recompensa_ppo > recompensa_reglas:
        conclusion = (
            "El agente PPO supero la recompensa promedio "
            "de la politica basada en reglas."
        )

    elif recompensa_ppo > recompensa_aleatorio:
        conclusion = (
            "PPO supero al agente aleatorio, pero obtuvo "
            "menor recompensa que la politica de reglas."
        )

    else:
        conclusion = (
            "PPO no supero al agente aleatorio. "
            "Se recomienda revisar su entrenamiento."
        )

    # ----------------------------------------
    # VERIFICAR EVIDENCIAS
    # ----------------------------------------

    evidencias = [
        "README.md",
        "requirements.txt",
        "src/p2_11_auditoria.py",
        "src/p2_12_entrega_final.py",
        "models/p2_ppo_robot.zip",
        "reports/p2_reporte_sesiones_6_9.md",
        "reports/p2_11_auditoria.json",
        "reports/figures/p2_cartpole_random.png",
        "reports/figures/p2_ppo_entrenamiento.png",
        "reports/figures/p2_09_recompensas.png",
        "reports/figures/p2_09_tasa_exito.png"
    ]

    inventario = []

    for nombre in evidencias:

        ruta = BASE / nombre

        inventario.append({
            "Archivo": nombre,
            "Disponible": ruta.is_file()
        })

    inventario = pd.DataFrame(inventario)

    ruta_inventario = (
        REPORTES / "p2_12_inventario.csv"
    )

    inventario.to_csv(
        ruta_inventario,
        index=False
    )

    entrega_lista = bool(
        auditoria.get("auditoria_aprobada", False)
        and inventario["Disponible"].all()
    )

    # ----------------------------------------
    # CREAR INFORME FINAL
    # ----------------------------------------

    tabla = tabla_markdown(comparacion)

    informe = f"""# Práctica 2 - Informe final

## 1. Objetivo general

Aplicar técnicas de evaluación, optimización y
aprendizaje por refuerzo mediante experimentos
reproducibles en Python.

## 2. Herramientas utilizadas

- Python
- pandas y NumPy
- scikit-learn
- Optuna
- Gymnasium
- Stable Baselines3
- Matplotlib
- Git y GitHub

## 3. Primera etapa: clasificación supervisada

Durante las sesiones 1 a 5 se desarrollaron
experimentos de clasificación desbalanceada
con el dataset Wall-Following Robot Navigation.

Se compararon clasificadores, se definió una
matriz de costos simulados, se ajustó el umbral
de decisión y se optimizaron hiperparámetros
de Random Forest mediante Optuna.

Los resultados detallados se encuentran en
el README y en los archivos de cada sesión.

## 4. Segunda etapa: aprendizaje por refuerzo

En la Sesión 6 se implementó un agente
aleatorio en CartPole-v1.

En la Sesión 7 se desarrolló RobotLineEnv,
un entorno personalizado de navegación lineal.

En la Sesión 8 se entrenó un agente PPO
utilizando Stable Baselines3.

En la Sesión 9 se compararon las políticas
aleatoria, basada en reglas y PPO.

## 5. Resultados de evaluación

Se utilizaron 30 episodios por política.

{tabla}

## 6. Interpretación

{conclusion}

La recompensa promedio de PPO fue
{recompensa_ppo:.2f}.

Su tasa de éxito fue
{ppo["tasa_exito"]:.2f}%.

Los resultados corresponden a un entorno
simplificado. No deben interpretarse como
validación en un robot físico.

## 7. Auditoría del proyecto

Fecha de auditoría:
{auditoria.get("fecha", "No disponible")}

Archivos verificados:
{auditoria.get("archivos_verificados", 0)}

Archivos presentes:
{auditoria.get("archivos_presentes", 0)}

Archivos faltantes:
{auditoria.get("archivos_faltantes", [])}

Modelo PPO:
{auditoria.get("modelo_ppo", {})}

Auditoría aprobada:
{auditoria.get("auditoria_aprobada", False)}

## 8. Evidencias

Las evidencias incluyen scripts, gráficos,
modelo PPO, reportes y documentación.

Inventario:
reports/p2_12_inventario.csv

## 9. Conclusión general

La práctica permitió integrar clasificación
supervisada, optimización de hiperparámetros
y aprendizaje por refuerzo.

Los resultados experimentales muestran el
comportamiento de diferentes políticas
dentro de un entorno de simulación controlado.

## 10. Estado de la entrega

Archivos y auditoría completos:
{entrega_lista}

Generado:
{datetime.now().isoformat(timespec="seconds")}
"""

    ruta_informe = (
        REPORTES / "p2_12_informe_final.md"
    )

    ruta_informe.write_text(
        informe,
        encoding="utf-8"
    )

    print("\n=== RESULTADOS FINALES ===")
    print(comparacion.to_string(index=False))

    print("\n=== VERIFICACION ===")
    print(inventario.to_string(index=False))

    print("\nAuditoria aprobada:",
          auditoria.get("auditoria_aprobada", False))

    print("Entrega lista:", entrega_lista)

    print("\n=== ARCHIVOS GENERADOS ===")
    print(ruta_inventario)
    print(ruta_informe)

    if entrega_lista:
        print("\nSesion 12 completada correctamente.")
    else:
        print(
            "\nSesion 12 genero el informe, "
            "pero existen pendientes."
        )


if __name__ == "__main__":
    main()

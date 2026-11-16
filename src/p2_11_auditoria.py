
from pathlib import Path
from datetime import datetime
import importlib.metadata as metadata
import json
import pandas as pd

from stable_baselines3 import PPO
from p2_07_robot_env import RobotLineEnv


# ============================================
# PRACTICA 2 - SESION 11
# AUDITORIA AUTOMATICA
# ============================================

BASE = Path(__file__).resolve().parent.parent
REPORTES = BASE / "reports"
DATOS = BASE / "data" / "processed"

REPORTES.mkdir(parents=True, exist_ok=True)

ARCHIVOS_OBLIGATORIOS = {
    "Sesión 1": [
        "notebooks/notebooks/p2_01_metricas.ipynb"
    ],
    "Sesión 2": ["src/p2_02_costos.py"],
    "Sesión 3": ["src/p2_03_threshold.py"],
    "Sesión 4": ["src/p2_04_optuna.py"],
    "Sesión 5": ["src/p2_05_optuna_analisis.py"],
    "Sesión 6": [
        "src/p2_06_cartpole_random.py",
        "reports/figures/p2_cartpole_random.png"
    ],
    "Sesión 7": [
        "src/p2_07_robot_env.py",
        "src/p2_07_test_env.py"
    ],
    "Sesión 8": [
        "src/p2_08_ppo_entrenamiento.py",
        "models/p2_ppo_robot.zip",
        "reports/figures/p2_ppo_entrenamiento.png"
    ],
    "Sesión 9": [
        "src/p2_09_evaluacion_ppo.py",
        "reports/figures/p2_09_recompensas.png",
        "reports/figures/p2_09_tasa_exito.png"
    ],
    "Sesión 10": [
        "src/p2_10_reporte.py",
        "reports/p2_reporte_sesiones_6_9.md"
    ]
}

RESULTADOS_ESPERADOS = [
    "p2_cartpole_random.csv",
    "p2_robot_env_resultados.csv",
    "p2_ppo_entrenamiento.csv",
    "p2_09_episodios.csv",
    "p2_09_comparacion.csv"
]

DEPENDENCIAS = [
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "gymnasium",
    "stable-baselines3",
    "optuna",
    "pyarrow"
]


def verificar_archivos():

    registros = []

    for sesion, archivos in ARCHIVOS_OBLIGATORIOS.items():

        for archivo in archivos:

            ruta = BASE / archivo

            registros.append({
                "Sesion": sesion,
                "Archivo": archivo,
                "Existe": ruta.is_file(),
                "Tamano_bytes": (
                    ruta.stat().st_size
                    if ruta.is_file() else 0
                )
            })

    return pd.DataFrame(registros)


def verificar_dependencias():

    resultados = []

    for paquete in DEPENDENCIAS:

        try:
            version = metadata.version(paquete)
            instalado = True
        except metadata.PackageNotFoundError:
            version = "No instalado"
            instalado = False

        resultados.append({
            "Paquete": paquete,
            "Instalado": instalado,
            "Version": version
        })

    return pd.DataFrame(resultados)


def verificar_modelo():

    ruta = BASE / "models" / "p2_ppo_robot.zip"

    if not ruta.exists():
        return {
            "carga_correcta": False,
            "error": "Modelo no encontrado"
        }

    try:
        modelo = PPO.load(
            str(ruta),
            device="cpu"
        )

        entorno = RobotLineEnv(
            meta=10,
            max_pasos=30
        )

        observacion, _ = entorno.reset(seed=42)

        recompensa_total = 0.0
        terminado = False
        truncado = False
        pasos = 0

        while not (terminado or truncado):

            accion, _ = modelo.predict(
                observacion,
                deterministic=True
            )

            observacion, recompensa, terminado, truncado, _ = (
                entorno.step(accion)
            )

            recompensa_total += recompensa
            pasos += 1

        entorno.close()

        return {
            "carga_correcta": True,
            "recompensa": recompensa_total,
            "pasos": pasos,
            "meta_alcanzada": bool(terminado)
        }

    except Exception as error:

        return {
            "carga_correcta": False,
            "error": str(error)
        }


def main():

    print("\n=== SESION 11: AUDITORIA ===")

    archivos = verificar_archivos()
    dependencias = verificar_dependencias()
    modelo = verificar_modelo()

    archivos_csv = []

    for nombre in RESULTADOS_ESPERADOS:

        ruta = DATOS / nombre

        archivos_csv.append({
            "Archivo": nombre,
            "Existe": ruta.exists()
        })

    resultados_csv = pd.DataFrame(archivos_csv)

    rutas_csv = {
        "archivos": REPORTES / "p2_11_archivos.csv",
        "dependencias": REPORTES / "p2_11_dependencias.csv",
        "resultados": REPORTES / "p2_11_resultados.csv"
    }

    archivos.to_csv(rutas_csv["archivos"], index=False)
    dependencias.to_csv(
        rutas_csv["dependencias"], index=False
    )
    resultados_csv.to_csv(
        rutas_csv["resultados"], index=False
    )

    resumen = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "archivos_verificados": len(archivos),
        "archivos_presentes": int(archivos["Existe"].sum()),
        "archivos_faltantes": archivos.loc[
            ~archivos["Existe"], "Archivo"
        ].tolist(),
        "dependencias_correctas": bool(
            dependencias["Instalado"].all()
        ),
        "resultados_csv_presentes": int(
            resultados_csv["Existe"].sum()
        ),
        "resultados_csv_esperados": len(resultados_csv),
        "modelo_ppo": modelo
    }

    resumen["auditoria_aprobada"] = bool(
        archivos["Existe"].all()
        and dependencias["Instalado"].all()
        and modelo["carga_correcta"]
        and modelo.get("meta_alcanzada", False)
        and resultados_csv["Existe"].all()
    )

    ruta_json = REPORTES / "p2_11_auditoria.json"

    ruta_json.write_text(
        json.dumps(
            resumen,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n=== ARCHIVOS ===")
    print(archivos.to_string(index=False))

    print("\n=== DEPENDENCIAS ===")
    print(dependencias.to_string(index=False))

    print("\n=== MODELO PPO ===")
    print(modelo)

    print("\n=== RESULTADOS ===")
    print(resultados_csv.to_string(index=False))

    print("\nAuditoria aprobada:",
          resumen["auditoria_aprobada"])

    print("\nReporte:", ruta_json)

    if not resumen["auditoria_aprobada"]:
        print(
            "\nATENCION: existen elementos pendientes."
        )
    else:
        print(
            "\nSesion 11 completada correctamente."
        )


if __name__ == "__main__":
    main()

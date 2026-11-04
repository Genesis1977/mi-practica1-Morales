
from pathlib import Path
import json

import optuna
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ============================================
# PRACTICA 2 - SESION 4: OPTUNA
# ============================================

BASE = Path(__file__).resolve().parent.parent

DATASET = BASE / "data" / "processed" / "dataset.parquet"
SALIDA = BASE / "data" / "processed"

N_TRIALS = 20
RANDOM_STATE = 42


def main():

    # ----------------------------------------
    # 1. CARGAR LOS DATOS
    # ----------------------------------------

    df = pd.read_parquet(DATASET)

    train = df[df["conjunto"] == "train"].copy()

    sensores = [
        f"sensor_{i}" for i in range(1, 25)
    ]

    X = train[sensores]

    clase_minoritaria = (
        train["clase"].value_counts().idxmin()
    )

    y = (
        train["clase"] == clase_minoritaria
    ).astype(int)

    print("\n=== SESION 4: OPTUNA ===")
    print("Clase minoritaria:", clase_minoritaria)
    print("Registros de entrenamiento:", len(X))

    # ----------------------------------------
    # 2. CONFIGURAR VALIDACION CRUZADA
    # ----------------------------------------

    validacion = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # ----------------------------------------
    # 3. DEFINIR FUNCION OBJETIVO
    # ----------------------------------------

    def objetivo(trial):

        parametros = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                50,
                200,
                step=25
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                5,
                30
            ),

            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                2,
                10
            ),

            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                1,
                5
            ),

            "max_features": trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2"]
            )
        }

        modelo = RandomForestClassifier(
            **parametros,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        puntuaciones = cross_val_score(
            modelo,
            X,
            y,
            cv=validacion,
            scoring="f1",
            n_jobs=1
        )

        return float(puntuaciones.mean())

    # ----------------------------------------
    # 4. CREAR O RECUPERAR ESTUDIO
    # ----------------------------------------

    SALIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    ruta_db = SALIDA / "p2_optuna.db"

    estudio = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        ),
        storage=f"sqlite:///{ruta_db.as_posix()}",
        study_name="p2_random_forest",
        load_if_exists=True
    )

    # ----------------------------------------
    # 5. CONTROLAR NUMERO DE PRUEBAS
    # ----------------------------------------

    completadas = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in estudio.trials
    )

    restantes = max(
        0,
        N_TRIALS - completadas
    )

    print("\n=== CONTROL DE PRUEBAS ===")
    print("Pruebas completadas:", completadas)
    print("Pruebas restantes:", restantes)

    if restantes > 0:

        print("\nEjecutando pruebas pendientes...")

        estudio.optimize(
            objetivo,
            n_trials=restantes
        )

    else:

        print(
            "\nEl estudio ya tiene suficientes pruebas."
        )

    # ----------------------------------------
    # 6. GUARDAR RESULTADOS
    # ----------------------------------------

    pruebas_finalizadas = [
        trial
        for trial in estudio.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    if not pruebas_finalizadas:
        raise RuntimeError(
            "No hay pruebas completadas en Optuna."
        )

    resultados = {
        "numero_trials": len(estudio.trials),
        "trials_completados": len(pruebas_finalizadas),
        "mejor_f1_validacion": estudio.best_value,
        "mejores_hiperparametros": estudio.best_params
    }

    ruta_json = (
        SALIDA / "p2_optuna_resultados.json"
    )

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

    ruta_csv = (
        SALIDA / "p2_optuna_trials.csv"
    )

    estudio.trials_dataframe().to_csv(
        ruta_csv,
        index=False
    )

    # ----------------------------------------
    # 7. MOSTRAR RESULTADOS
    # ----------------------------------------

    print("\n=== RESULTADOS DE OPTUNA ===")

    print(
        "Pruebas registradas:",
        len(estudio.trials)
    )

    print(
        "Pruebas completadas:",
        len(pruebas_finalizadas)
    )

    print(
        f"Mejor F1 de validacion: "
        f"{estudio.best_value:.4f}"
    )

    print("\nMejores hiperparametros:")

    for nombre, valor in estudio.best_params.items():
        print(f"  {nombre}: {valor}")

    print("\n=== ARCHIVOS GENERADOS ===")
    print(ruta_db)
    print(ruta_json)
    print(ruta_csv)

    print("\nSesion 4 completada correctamente.")


if __name__ == "__main__":
    main()


from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

# ============================================
# PRACTICA 2 - SESION 2: MATRIZ DE COSTOS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR / "data" / "processed" / "dataset.parquet"
)

COSTO_FP = 50
COSTO_FN = 500


def evaluar_costos(y_real, y_predicho):
    """Calcula la matriz de confusion y el costo simulado."""

    tn, fp, fn, tp = confusion_matrix(
        y_real,
        y_predicho,
        labels=[0, 1]
    ).ravel()

    costo_total = fp * COSTO_FP + fn * COSTO_FN

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Costo total ($)": int(costo_total)
    }


def main():
    """Entrena y compara el costo de ambos clasificadores."""

    # 1. Cargar datos procesados de la Practica 1
    df = pd.read_parquet(DATASET_PATH)

    # 2. Recuperar division original
    train = df[df["conjunto"] == "train"].copy()
    test = df[df["conjunto"] == "test"].copy()

    sensores = [f"sensor_{i}" for i in range(1, 25)]

    X_train = train[sensores]
    X_test = test[sensores]

    # 3. Seleccionar clase minoritaria
    clase_minoritaria = train["clase"].value_counts().idxmin()

    y_train = (
        train["clase"] == clase_minoritaria
    ).astype(int)

    y_test = (
        test["clase"] == clase_minoritaria
    ).astype(int)

    print("Clase minoritaria:", clase_minoritaria)

    # 4. Crear modelos de la Sesion 1
    modelos = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "Modelo trivial": DummyClassifier(
            strategy="constant",
            constant=0
        )
    }

    resultados = []

    # 5. Entrenar y evaluar cada modelo
    for nombre, modelo in modelos.items():

        modelo.fit(X_train, y_train)

        predicciones = modelo.predict(X_test)

        resultado = evaluar_costos(
            y_test,
            predicciones
        )

        resultados.append({
            "Modelo": nombre,
            **resultado
        })

    # 6. Mostrar tabla comparativa
    tabla = pd.DataFrame(resultados)

    print("\n=== MATRIZ DE COSTOS ===")
    print(tabla.to_string(index=False))

    # 7. Calcular ahorro economico
    costo_rf = tabla.loc[
        tabla["Modelo"] == "Random Forest",
        "Costo total ($)"
    ].iloc[0]

    costo_trivial = tabla.loc[
        tabla["Modelo"] == "Modelo trivial",
        "Costo total ($)"
    ].iloc[0]

    ahorro = costo_trivial - costo_rf

    print("\n=== COMPARACION ECONOMICA ===")
    print(f"Costo Random Forest: ${costo_rf:,.2f}")
    print(f"Costo modelo trivial: ${costo_trivial:,.2f}")
    print(f"Ahorro simulado: ${ahorro:,.2f}")

    if costo_trivial > 0:
        porcentaje = (ahorro / costo_trivial) * 100

        print(
            f"Reduccion de costos: {porcentaje:.2f}%"
        )

    # 8. Guardar tabla para documentacion
    salida = (
        BASE_DIR / "data" / "processed" /
        "p2_comparacion_costos.csv"
    )

    tabla.to_csv(salida, index=False)

    print("\nArchivo generado:", salida)


if __name__ == "__main__":
    main()


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# ============================================
# PRACTICA 2 - SESION 3: THRESHOLD TUNING
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = BASE_DIR / "data" / "processed" / "dataset.parquet"
FIGURES = BASE_DIR / "reports" / "figures"
PROCESSED = BASE_DIR / "data" / "processed"

FIGURES.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

COSTO_FP = 50
COSTO_FN = 500
RANDOM_STATE = 42


def calcular_costo(y_real, y_pred):
    """Calcula FP, FN y costo economico simulado."""

    tn, fp, fn, tp = confusion_matrix(
        y_real,
        y_pred,
        labels=[0, 1]
    ).ravel()

    costo_total = fp * COSTO_FP + fn * COSTO_FN

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Costo ($)": int(costo_total)
    }


def evaluar_umbral(y_real, probabilidades, umbral):
    """Evalua las predicciones usando un umbral."""

    predicciones = (
        probabilidades >= umbral
    ).astype(int)

    return calcular_costo(y_real, predicciones)


def main():
    """Optimiza el umbral en validacion y evalua en test."""

    # ----------------------------------------
    # 1. CARGAR DATASET
    # ----------------------------------------

    df = pd.read_parquet(DATASET)

    train = df[df["conjunto"] == "train"].copy()
    test = df[df["conjunto"] == "test"].copy()

    sensores = [
        f"sensor_{i}" for i in range(1, 25)
    ]

    # Identificar clase minoritaria
    clase_minoritaria = (
        train["clase"].value_counts().idxmin()
    )

    X_train_original = train[sensores]
    y_train_original = (
        train["clase"] == clase_minoritaria
    ).astype(int)

    X_test = test[sensores]
    y_test = (
        test["clase"] == clase_minoritaria
    ).astype(int)

    print("\n=== SESION 3: THRESHOLD TUNING ===")
    print("Clase minoritaria:", clase_minoritaria)

    # ----------------------------------------
    # 2. DIVIDIR TRAIN EN TRAIN Y VALIDACION
    # ----------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_original,
        y_train_original,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_original
    )

    print("\n=== DISTRIBUCION DE DATOS ===")
    print("Entrenamiento:", len(X_train))
    print("Validacion:", len(X_val))
    print("Prueba:", len(X_test))

    # ----------------------------------------
    # 3. ENTRENAR RANDOM FOREST
    # ----------------------------------------

    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    modelo.fit(X_train, y_train)

    # Probabilidades de validacion
    probabilidades_val = (
        modelo.predict_proba(X_val)[:, 1]
    )

    # ----------------------------------------
    # 4. PROBAR UMBRALES EN VALIDACION
    # ----------------------------------------

    resultados = []

    umbrales = np.round(
        np.arange(0.10, 0.901, 0.05),
        2
    )

    for umbral in umbrales:

        metricas = evaluar_umbral(
            y_val,
            probabilidades_val,
            umbral
        )

        resultados.append({
            "Umbral": float(umbral),
            **metricas
        })

    tabla = pd.DataFrame(resultados)

    # Seleccionar el menor costo en validacion
    mejor = tabla.loc[
        tabla["Costo ($)"].idxmin()
    ]

    umbral_optimo = float(mejor["Umbral"])

    base_val = tabla[
        np.isclose(tabla["Umbral"], 0.50)
    ].iloc[0]

    print("\n=== RESULTADOS DE VALIDACION ===")
    print(tabla.to_string(index=False))

    print("\n=== UMBRAL SELECCIONADO ===")
    print(f"Umbral optimo: {umbral_optimo:.2f}")
    print(
        "Costo validacion con umbral 0.50:",
        f"${int(base_val['Costo ($)']):,}"
    )
    print(
        "Costo minimo en validacion:",
        f"${int(mejor['Costo ($)']):,}"
    )

    # ----------------------------------------
    # 5. EVALUAR EN TEST
    # ----------------------------------------

    # Reentrenar con todo el train original
    # sin utilizar el conjunto de prueba.
    modelo_final = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    modelo_final.fit(
        X_train_original,
        y_train_original
    )

    probabilidades_test = (
        modelo_final.predict_proba(X_test)[:, 1]
    )

    # Umbral tradicional 0.50
    resultado_base = evaluar_umbral(
        y_test,
        probabilidades_test,
        0.50
    )

    # Umbral elegido en validacion
    resultado_optimo = evaluar_umbral(
        y_test,
        probabilidades_test,
        umbral_optimo
    )

    ahorro_test = (
        resultado_base["Costo ($)"]
        - resultado_optimo["Costo ($)"]
    )

    print("\n=== EVALUACION FINAL EN TEST ===")

    comparacion = pd.DataFrame([
        {
            "Modelo": "Umbral tradicional",
            "Umbral": 0.50,
            **resultado_base
        },
        {
            "Modelo": "Umbral optimizado",
            "Umbral": umbral_optimo,
            **resultado_optimo
        }
    ])

    print(comparacion.to_string(index=False))

    print("\n=== RESULTADO FINAL ===")
    print(f"Umbral seleccionado: {umbral_optimo:.2f}")
    print(
        "Costo test con umbral 0.50:",
        f"${resultado_base['Costo ($)']:,}"
    )
    print(
        "Costo test con umbral optimizado:",
        f"${resultado_optimo['Costo ($)']:,}"
    )
    print(f"Ahorro simulado en test: ${ahorro_test:,}")

    # ----------------------------------------
    # 6. GENERAR GRAFICA DE VALIDACION
    # ----------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        tabla["Umbral"],
        tabla["Costo ($)"],
        marker="o",
        label="Costo en validacion"
    )

    plt.axvline(
        0.50,
        linestyle="--",
        label="Umbral tradicional (0.50)"
    )

    plt.axvline(
        umbral_optimo,
        linestyle=":",
        label=f"Umbral seleccionado ({umbral_optimo:.2f})"
    )

    plt.scatter(
        [umbral_optimo],
        [mejor["Costo ($)"]],
        s=100,
        zorder=5,
        label="Costo minimo en validacion"
    )

    plt.title(
        "Threshold Tuning - Costo vs Umbral (Validacion)"
    )
    plt.xlabel("Umbral de decision")
    plt.ylabel("Costo economico simulado ($)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    ruta_grafico = FIGURES / "threshold_tuning.png"

    plt.savefig(
        ruta_grafico,
        dpi=300
    )

    plt.close()

    # ----------------------------------------
    # 7. GUARDAR RESULTADOS
    # ----------------------------------------

    ruta_validacion = (
        PROCESSED / "p2_threshold_validacion.csv"
    )

    ruta_comparacion = (
        PROCESSED / "p2_threshold_comparacion.csv"
    )

    tabla.to_csv(
        ruta_validacion,
        index=False
    )

    comparacion.to_csv(
        ruta_comparacion,
        index=False
    )

    print("\n=== ARCHIVOS GENERADOS ===")
    print(ruta_grafico)
    print(ruta_validacion)
    print(ruta_comparacion)

    print("\nSesion 3 completada correctamente.")


if __name__ == "__main__":
    main()

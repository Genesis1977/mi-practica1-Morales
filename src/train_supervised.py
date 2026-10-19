
import joblib
import pandas as pd
from pathlib import Path
from time import perf_counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from data_loader import load_raw_data, preprocess, split_data


# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "sensor_readings_24.data"
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# MODELO 1: RANDOM FOREST
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


# MODELO 2: KNN
def train_knn(X_train, y_train):
    model = KNeighborsClassifier(
        n_neighbors=5
    )

    model.fit(X_train, y_train)
    return model


# MODELO 3: SVM
def train_svm(X_train, y_train):
    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


# EVALUACION DE MODELOS
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    resultados = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
            labels=model.classes_
        )
    }

    return resultados


# MOSTRAR RESULTADOS
def mostrar_resultados(nombre, resultados, clases):
    print(f"\n========== {nombre} ==========")

    print("Accuracy:", round(resultados["accuracy"], 4))
    print("F1-score:", round(resultados["f1_score"], 4))

    print("\nOrden de las clases:")
    print(list(clases))

    print("\nMatriz de confusion:")
    print(resultados["confusion_matrix"])


# PROGRAMA PRINCIPAL
if __name__ == "__main__":

    # 1. Preparar carpetas
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Cargar datos
    df = load_raw_data(DATA_PATH)

    # 3. Separar X e y
    X, y = preprocess(df)

    # 4. Dividir y escalar sin Data Leakage
    X_train, X_test, y_train, y_test, scaler = split_data(X, y)

    print("Datos preparados correctamente.")
    print("Entrenamiento:", X_train.shape)
    print("Prueba:", X_test.shape)

    
    # 5. Entrenar los tres modelos y medir tiempos

    print("\nEntrenando Random Forest...")
    inicio = perf_counter()
    random_forest = train_random_forest(X_train, y_train)
    tiempo_rf = perf_counter() - inicio

    print("\nEntrenando KNN...")
    inicio = perf_counter()
    knn = train_knn(X_train, y_train)
    tiempo_knn = perf_counter() - inicio

    print("\nEntrenando SVM...")
    inicio = perf_counter()
    svm = train_svm(X_train, y_train)
    tiempo_svm = perf_counter() - inicio

    tiempos_entrenamiento = {
        "Random Forest": tiempo_rf,
        "KNN": tiempo_knn,
        "SVM": tiempo_svm
    }

    print("\nTodos los modelos fueron entrenados.")

    print("\nTiempos de entrenamiento:")
    for nombre, tiempo in tiempos_entrenamiento.items():
        print(f"{nombre}: {tiempo:.4f} segundos")

    print("\nTodos los modelos fueron entrenados.")

    # 6. Evaluar los modelos
    modelos = {
        "Random Forest": random_forest,
        "KNN": knn,
        "SVM": svm
    }

    resultados_comparacion = []

    for nombre, modelo in modelos.items():

        resultados = evaluate_model(
            modelo,
            X_test,
            y_test
        )

        mostrar_resultados(
            nombre,
            resultados,
            modelo.classes_
        )

        resultados_comparacion.append({
            "Modelo": nombre,
            "Accuracy": resultados["accuracy"],
            "F1-score": resultados["f1_score"],
            "Tiempo (s)": tiempos_entrenamiento[nombre]
        })

    # 7. Crear tabla comparativa
    comparacion = pd.DataFrame(resultados_comparacion)

    comparacion = comparacion.sort_values(
        by="F1-score",
        ascending=False
    )

    print("\n========== COMPARACION FINAL ==========")
    print(comparacion.to_string(index=False))

    # 8. Identificar el mejor modelo segun F1
    mejor_modelo = comparacion.iloc[0]["Modelo"]

    print("\nMejor modelo segun F1-score:", mejor_modelo)

    # 9. Guardar modelos
    joblib.dump(
        random_forest,
        MODELS_DIR / "random_forest.pkl"
    )

    joblib.dump(
        knn,
        MODELS_DIR / "knn.pkl"
    )

    joblib.dump(
        svm,
        MODELS_DIR / "svm.pkl"
    )

    joblib.dump(
        scaler,
        MODELS_DIR / "robust_scaler.pkl"
    )

    # 10. Guardar resultados para el notebook
    comparacion.to_csv(
        PROCESSED_DIR / "comparacion_modelos.csv",
        index=False
    )

    print("\nArchivos guardados correctamente:")
    print("- models/random_forest.pkl")
    print("- models/knn.pkl")
    print("- models/svm.pkl")
    print("- models/robust_scaler.pkl")
    print("- data/processed/comparacion_modelos.csv")

    print("\nClase 6: entrenamiento y comparacion finalizados.")

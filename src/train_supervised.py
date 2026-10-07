import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from data_loader import load_raw_data, preprocess, split_data


def train_random_forest(X_train, y_train):
    """
    Entrena un modelo Random Forest para clasificar
    las acciones de navegación del robot.
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo utilizando el conjunto de prueba.
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    matriz_confusion = confusion_matrix(
        y_test,
        y_pred
    )

    resultados = {
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": matriz_confusion
    }

    return resultados

if __name__ == "__main__":

    # 1. Cargar el dataset original
    df = load_raw_data("../data/raw/sensor_readings_24.data")

    # 2. Separar variables predictoras y variable objetivo
    X, y = preprocess(df)

    # 3. Dividir los datos y aplicar el escalado
    X_train, X_test, y_train, y_test, scaler = split_data(X, y)

    print("Datos preparados correctamente.")
    print("Entrenamiento:", X_train.shape)
    print("Prueba:", X_test.shape)

    # 4. Entrenar Random Forest
    print("\nEntrenando Random Forest...")

    model = train_random_forest(
        X_train,
        y_train
    )

    print("Modelo entrenado correctamente.")

    # 5. Evaluar el modelo
    resultados = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\nResultados del modelo:")
    print("Accuracy:", round(resultados["accuracy"], 4))
    print("F1-score:", round(resultados["f1_score"], 4))

    print("\nMatriz de confusión:")
    print(resultados["confusion_matrix"])

    # 6. Guardar el modelo entrenado
    joblib.dump(
        model,
        "../models/random_forest.pkl"
    )

    print("\nModelo guardado en: models/random_forest.pkl")
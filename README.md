# Práctica 1 – Machine Learning aplicado a Robótica

**Estudiante:** Genesis Morales  
**Asignatura:** Machine Learning – RIA-805  
**Carrera:** Ingeniería en Robótica e Inteligencia Artificial  
**Universidad:** Universidad Católica de Cuenca

## 1. Descripción del proyecto

Este proyecto desarrolla un pipeline de Machine Learning aplicado al análisis de datos de sensores de un robot móvil.

Se utilizan técnicas de análisis exploratorio de datos (EDA), preprocesamiento, aprendizaje supervisado, aprendizaje no supervisado, evaluación de modelos y visualización de resultados.

El proyecto está organizado en nueve clases prácticas que permiten desarrollar un flujo de trabajo reproducible utilizando Python, scikit-learn, Jupyter Notebook y GitHub.

## 2. Dataset utilizado

**Nombre:** Wall-Following Robot Navigation  
**Fuente:** UCI Machine Learning Repository  
**Archivo:** `sensor_readings_24.data`

El conjunto de datos contiene:

- 5456 registros.
- 24 variables numéricas correspondientes a lecturas de sensores.
- Una variable objetivo que representa cuatro clases de navegación.

El objetivo es analizar los patrones de los sensores y aplicar algoritmos de clasificación y agrupamiento.

## 3. Objetivos

### Objetivo general

Desarrollar un pipeline de Machine Learning para procesar, analizar y modelar datos de sensores robóticos mediante técnicas de aprendizaje supervisado y no supervisado.

### Objetivos específicos

1. Realizar un análisis exploratorio para comprender las características del dataset.
2. Implementar un proceso de preparación, división y escalado de los datos.
3. Entrenar y evaluar algoritmos de clasificación supervisada.
4. Implementar técnicas de agrupamiento no supervisado.
5. Visualizar los resultados mediante PCA y t-SNE.
6. Medir la latencia de inferencia y documentar los resultados obtenidos.

## 4. Estructura del proyecto

```text
mi-practica1-Morales/
│
├── data/
│   ├── raw/
│   │   └── sensor_readings_24.data
│   └── processed/
│       ├── dataset.parquet
│       ├── eda_summary.json
│       ├── comparacion_modelos.csv
│       ├── metodo_codo.csv
│       ├── metodo_codo.png
│       ├── comparacion_clustering.csv
│       └── benchmark_latency.json
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_supervised.ipynb
│   └── 03_unsupervised.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── train_supervised.py
│   └── train_unsupervised.py
│
├── models/
│   ├── random_forest.pkl
│   ├── knn.pkl
│   ├── svm.pkl
│   ├── robust_scaler.pkl
│   ├── kmeans.pkl
│   ├── dbscan.pkl
│   └── unsupervised_scaler.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

Los archivos de resultados y modelos se generan durante la ejecución de los programas. Su disponibilidad en GitHub depende de la configuración de `.gitignore`.

## 5. Tecnologías utilizadas

- Python: lenguaje principal de programación.
- Pandas y NumPy: manipulación y procesamiento de datos.
- Scikit-learn: entrenamiento y evaluación de algoritmos.
- Matplotlib: generación de visualizaciones.
- Joblib: almacenamiento de modelos entrenados.
- PyArrow: almacenamiento de datos en formato Parquet.
- Jupyter Notebook: experimentación y presentación de resultados.
- Git y GitHub: control de versiones.

## 6. Instalación y configuración

### Clonar el repositorio

```bash
git clone https://github.com/Genesis1977/mi-practica1-Morales.git
cd mi-practica1-Morales
```

### Crear un entorno virtual

```bash
python -m venv venv
```

### Activar el entorno virtual en Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

### Instalar las dependencias

```bash
python -m pip install -r requirements.txt
```

### Ejecutar los programas

Preparación de datos:

```bash
python src/data_loader.py
```

Entrenamiento supervisado:

```bash
python src/train_supervised.py
```

Entrenamiento no supervisado:

```bash
python src/train_unsupervised.py
```

El programa no supervisado muestra la gráfica del método del codo y solicita seleccionar un valor de `k` entre 2 y 10.

Para reproducir los experimentos realizados se seleccionó `k=4`.

Los notebooks deben abrirse en VS Code o Jupyter utilizando el entorno de Python configurado. Se recomienda ejecutar sus celdas en orden.

## 7. Desarrollo de las clases

| Clase | Actividad |
|---|---|
| 1 | Configuración del repositorio y estructura del proyecto |
| 2 | Carga y exploración inicial del dataset |
| 3 | Análisis exploratorio de datos y estadísticas |
| 4 | Preprocesamiento, división estratificada y escalado |
| 5 | Entrenamiento y evaluación de Random Forest |
| 6 | Comparación de Random Forest, KNN y SVM |
| 7 | Agrupamiento mediante K-Means y DBSCAN |
| 8 | Visualizaciones PCA, t-SNE y benchmark de latencia |
| 9 | Documentación, reproducibilidad y entrega final |

## 8. Preprocesamiento

Se implementó un proceso que incluye:

1. Carga del conjunto de datos original.
2. Separación de las 24 variables predictoras y la variable objetivo.
3. División estratificada del 80 % para entrenamiento y 20 % para prueba.
4. Aplicación de RobustScaler.
5. Almacenamiento del dataset procesado en formato Parquet.

Para los modelos supervisados, el escalador se ajusta únicamente con los datos de entrenamiento y después transforma los datos de prueba. Este procedimiento evita la fuga de información (*data leakage*) durante el preprocesamiento.

## 9. Aprendizaje supervisado

Se implementaron tres algoritmos de clasificación:

- Random Forest.
- K-Nearest Neighbors (KNN).
- Support Vector Machine (SVM).

### Resultados obtenidos

| Modelo | Accuracy | F1-score ponderado |
|---|---:|---:|
| Random Forest | 0.9927 | 0.9927 |
| KNN | 0.8791 | 0.8789 |
| SVM | 0.8690 | 0.8684 |

Random Forest presentó el mejor rendimiento predictivo de los tres modelos evaluados.

También se generaron matrices de confusión, una comparación gráfica y mediciones del tiempo de entrenamiento.

Los resultados completos, incluidos los tiempos medidos, se encuentran en `data/processed/comparacion_modelos.csv`.

## 10. Aprendizaje no supervisado

Se implementaron dos algoritmos de agrupamiento:

**K-Means:** permite dividir los datos en un número determinado de clústeres. Se aplicó el método del codo para estudiar diferentes valores de `k`.

**DBSCAN:** permite identificar agrupamientos basados en densidad y registros considerados ruido.

### Resultados obtenidos

| Métrica | K-Means | DBSCAN |
|---|---:|---:|
| Clústeres | 4 | 32 |
| Silhouette Score | 0.4709 | 0.1834 |
| Registros considerados ruido | 0 | 849 |

K-Means obtuvo un Silhouette Score superior en la configuración evaluada. DBSCAN identificó 849 registros considerados ruido, equivalentes aproximadamente al 15,56 % del dataset.

La comparación de Silhouette Score es orientativa, porque DBSCAN fue evaluado excluyendo los registros considerados ruido.

## 11. Visualización mediante PCA y t-SNE

Se utilizaron dos técnicas de reducción de dimensionalidad:

**PCA:** transforma los datos en componentes principales para representar su variabilidad en dos dimensiones.

**t-SNE:** genera una representación bidimensional orientada a explorar relaciones de proximidad entre observaciones.

Ambas técnicas permitieron visualizar y comparar los agrupamientos obtenidos mediante K-Means y DBSCAN.

Las visualizaciones se encuentran en `notebooks/03_unsupervised.ipynb`.

## 12. Benchmark de latencia

Se implementó una prueba de rendimiento para medir el tiempo de inferencia del modelo K-Means.

El procedimiento consistió en:

1. Seleccionar una observación del dataset procesado.
2. Realizar cinco predicciones de calentamiento.
3. Ejecutar 100 predicciones individuales.
4. Medir los tiempos mediante `time.perf_counter()`.
5. Calcular la latencia promedio, mínima y máxima.
6. Guardar los resultados en `benchmark_latency.json`.

Esta prueba mide la latencia de inferencia de K-Means en el entorno utilizado. No representa el tiempo total de respuesta de un robot físico.

## 13. Reproducibilidad

Para reproducir los experimentos se recomienda:

- Utilizar las dependencias indicadas en `requirements.txt`.
- Ejecutar los comandos desde la carpeta principal del proyecto.
- Mantener el dataset original sin modificaciones.
- Conservar `random_state=42` en los algoritmos que lo utilizan.
- Ejecutar los scripts antes de los notebooks que dependen de modelos guardados.
- Utilizar `k=4` y `eps=3.0` para reproducir la configuración no supervisada evaluada.

Los tiempos de entrenamiento e inferencia pueden variar dependiendo del equipo y del entorno de ejecución.

## 14. Conclusiones generales

Se desarrolló un pipeline de Machine Learning aplicado a datos reales de sensores robóticos, integrando las principales etapas de preparación, entrenamiento, evaluación y documentación.

En el aprendizaje supervisado, Random Forest presentó el mejor rendimiento predictivo entre los tres algoritmos evaluados, con un Accuracy y F1-score ponderado aproximados del 99,27 %.

En el aprendizaje no supervisado, K-Means permitió generar cuatro agrupamientos con una separación moderada, mientras que DBSCAN identificó estructuras basadas en densidad y registros considerados ruido.

Las técnicas PCA y t-SNE facilitaron la exploración visual de los resultados. Además, el benchmark de latencia permitió incorporar una medición básica del costo de inferencia de K-Means.

El proyecto demuestra la aplicación de técnicas de Machine Learning a datos de robótica. Los resultados corresponden a experimentos con un dataset y no constituyen una validación del sistema en un robot físico.

## 15. Referencia del dataset

UCI Machine Learning Repository. *Wall-Following Robot Navigation Data Set*.

https://archive.ics.uci.edu/dataset/194/wall+following+robot+navigation+data

## 16. Repositorio

GitHub: https://github.com/Genesis1977/mi-practica1-Morales

## Práctica 2 — Sesión 2: Matriz de costos

En esta sesión se evalúa el impacto económico simulado de los errores de clasificación de Random Forest y de un modelo trivial, utilizando el problema binario definido en la Sesión 1.

Se asigna un costo de **$50 por falso positivo (FP)** y **$500 por falso negativo (FN)**, conforme a los valores establecidos en la actividad.

Estos costos representan un escenario hipotético de navegación robótica. Un falso positivo podría provocar una respuesta preventiva innecesaria, mientras que un falso negativo podría representar la omisión de una condición relevante para la navegación. Por esta razón, se considera que el falso negativo tiene un impacto diez veces mayor.

El costo total se calcula mediante:

**Costo total = (FP × $50) + (FN × $500)**

La comparación permite determinar si Random Forest reduce los costos simulados respecto a un clasificador trivial que siempre predice la clase negativa.

## Práctica 2 — Evaluación, optimización y aprendizaje por refuerzo

### Descripción

La Práctica 2 continúa el trabajo realizado con el dataset *Wall-Following Robot Navigation*, incorporando evaluación de modelos en escenarios desbalanceados, análisis de costos, ajuste del umbral de clasificación y optimización de hiperparámetros.

Las sesiones 1 a 5 utilizan Random Forest y un clasificador trivial para estudiar el impacto de los errores de predicción. Las siguientes sesiones abordarán aprendizaje por refuerzo con Gymnasium.

### Sesión 1 — Métricas para clases desbalanceadas

Se transformó el problema original en una clasificación binaria, considerando como clase positiva a `Slight-Left-Turn`, identificada como clase minoritaria.

Se compararon Random Forest y un modelo trivial que predice siempre la clase negativa.

El modelo trivial obtuvo aproximadamente 93,96 % de Accuracy, pero un F1-score de 0 para la clase positiva. Esto demuestra que una exactitud elevada no garantiza un buen desempeño cuando las clases están desbalanceadas.

Se utilizaron Accuracy, Precision, Recall, F1-score y matriz de confusión para evaluar los modelos.

**Archivo:** `notebooks/notebooks/p2_01_metricas.ipynb`

### Sesión 2 — Matriz de costos

Se implementó una matriz de costos para evaluar las consecuencias simuladas de los errores de clasificación.

Se asignaron los siguientes valores:

- Falso positivo (FP): $50.
- Falso negativo (FN): $500.

**Fórmula:**

Costo total = (FP × 50) + (FN × 500)

Los costos representan supuestos académicos para un escenario hipotético de navegación robótica. No corresponden a costos económicos reales medidos en un robot.

Se asignó un costo mayor a los falsos negativos porque representan situaciones positivas que el clasificador no consigue detectar.

**Archivo:** `src/p2_02_costos.py`

### Sesión 3 — Threshold Tuning

Se implementó un procedimiento de ajuste del umbral de decisión para reducir el costo simulado de los errores de clasificación.

Los datos originales de entrenamiento se dividieron nuevamente en entrenamiento y validación mediante una partición estratificada.

Se evaluaron umbrales entre 0.10 y 0.90, con incrementos de 0.05. El umbral de menor costo se seleccionó utilizando exclusivamente el conjunto de validación.

Posteriormente, se entrenó un modelo con todo el conjunto original de entrenamiento y se evaluó el umbral seleccionado sobre el conjunto de prueba.

Este procedimiento evita utilizar los datos de prueba para seleccionar el umbral, reduciendo el riesgo de fuga de información.

**Archivos:**

- `src/p2_03_threshold.py`
- `reports/figures/threshold_tuning.png`
- `data/processed/p2_threshold_validacion.csv`
- `data/processed/p2_threshold_comparacion.csv`

### Sesión 4 — Optimización de hiperparámetros con Optuna

Se utilizó Optuna para optimizar los hiperparámetros de Random Forest mediante 20 pruebas y validación cruzada estratificada de tres particiones.

Los hiperparámetros evaluados fueron `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` y `max_features`.

**Resultados:**

- Pruebas completadas: 20.
- Mejor F1 promedio de validación cruzada: 0.9580.
- Número de árboles: 175.
- Profundidad máxima: 8.
- Mínimo de muestras para dividir: 6.
- Mínimo de muestras por hoja: 1.
- Variables consideradas por división: `log2`.

El estudio se almacena mediante SQLite y permite recuperar las pruebas existentes sin ejecutar nuevas pruebas innecesariamente.

**Archivo:** `src/p2_04_optuna.py`

### Sesión 5 — Análisis y comparación de modelos

Se analizaron los resultados obtenidos mediante Optuna, incluyendo el historial de optimización y la importancia relativa de los hiperparámetros.

Se compararon Random Forest original y Random Forest optimizado sobre el conjunto de prueba.

| Métrica | Random Forest original | Random Forest Optuna |
|---|---:|---:|
| Accuracy | 0.996337 | 0.997253 |
| Precision | 0.984375 | 0.984615 |
| Recall | 0.954545 | 0.969697 |
| F1-score | 0.969231 | 0.977099 |
| Falsos positivos | 1 | 1 |
| Falsos negativos | 3 | 2 |
| Costo simulado | $1,550 | $1,050 |

El modelo optimizado mejoró el F1-score y redujo de tres a dos los falsos negativos, manteniendo un falso positivo.

La reducción del costo simulado fue de aproximadamente 32.26 %, equivalente a $500.

Estos resultados corresponden a la evaluación experimental del dataset y no constituyen una validación sobre un robot físico.

**Archivos:**

- `src/p2_05_optuna_analisis.py`
- `config.json`
- `reports/figures/optuna_historial.png`
- `reports/figures/optuna_parametros.png`

### Ejecución de las sesiones

Los siguientes comandos deben ejecutarse desde la carpeta principal del repositorio, con el entorno virtual y las dependencias instaladas.

```powershell
.\venv\Scripts\python.exe src\p2_02_costos.py
.\venv\Scripts\python.exe src\p2_03_threshold.py
.\venv\Scripts\python.exe src\p2_04_optuna.py
.\venv\Scripts\python.exe src\p2_05_optuna_analisis.py
```

La Sesión 1 se ejecuta desde su notebook utilizando el intérprete de Python del entorno virtual.

Los scripts requieren el archivo `data/processed/dataset.parquet`, generado durante la Práctica 1.

Para reproducir la optimización de Optuna se recomienda ejecutar primero la Sesión 4 y después la Sesión 5.

### Estado del proyecto

Las sesiones 1 a 5 de la Práctica 2 se encuentran implementadas. Se comprobó la ejecución correcta de las sesiones 4 y 5, incluyendo la persistencia de las 20 pruebas de Optuna y la comparación final de modelos.

La siguiente etapa corresponde a la Sesión 6, dedicada a la introducción al aprendizaje por refuerzo mediante Gymnasium y el entorno CartPole.
# Práctica 2 - Informe final

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

| Politica | recompensa_promedio | pasos_promedio | tasa_exito |
| --- | --- | --- | --- |
| Aleatorio | -22.6 | 29.33 | 6.67 |
| Reglas | 91.0 | 10.0 | 100.0 |
| PPO | 91.0 | 10.0 | 100.0 |

## 6. Interpretación

El agente PPO igualo la recompensa promedio de la politica basada en reglas.

La recompensa promedio de PPO fue
91.00.

Su tasa de éxito fue
100.00%.

Los resultados corresponden a un entorno
simplificado. No deben interpretarse como
validación en un robot físico.

## 7. Auditoría del proyecto

Fecha de auditoría:
2026-11-16T12:46:27

Archivos verificados:
17

Archivos presentes:
17

Archivos faltantes:
[]

Modelo PPO:
{'carga_correcta': True, 'recompensa': 91.0, 'pasos': 10, 'meta_alcanzada': True}

Auditoría aprobada:
True

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
True

Generado:
2026-11-16T12:46:38

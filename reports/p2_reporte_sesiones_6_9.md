# Practica 2 - Reporte de aprendizaje por refuerzo

## Objetivo

Implementar y evaluar agentes de aprendizaje por
refuerzo utilizando Gymnasium y Stable Baselines3.

## Sesion 6 - CartPole

Se ejecuto un agente aleatorio durante cinco episodios.

Recompensa promedio: 37.40.

El agente no fue entrenado y se utilizo como
referencia de comportamiento aleatorio en CartPole.

## Sesion 7 - Entorno personalizado

Se implemento RobotLineEnv, un entorno donde un
robot debe alcanzar una posicion objetivo.

Resultados:

           recompensa  pasos  exito
Politica                           
aleatoria        15.0   25.4    0.4
reglas           91.0   10.0    1.0

## Sesion 8 - Entrenamiento PPO

Se entreno un agente PPO con 10000 pasos solicitados,
utilizando MlpPolicy en RobotLineEnv.


Episodios registrados: 838.

Recompensa media de los primeros 20 episodios:
-2.55.

Recompensa media de los últimos 20 episodios:
90.50.

Estos valores describen la evolución observada
durante el entrenamiento. La evaluación del
agente se presenta en la Sesión 9.


## Sesion 9 - Evaluacion comparativa

Se evaluaron 30 episodios por politica.

 Politica  recompensa_promedio  recompensa_std  pasos_promedio  tasa_exito
Aleatorio                -22.6       28.239859       29.333333    6.666667
   Reglas                 91.0        0.000000       10.000000  100.000000
      PPO                 91.0        0.000000       10.000000  100.000000

Recompensa media PPO: 91.00.

Tasa de exito PPO: 100.00%.

## Conclusion

PPO alcanzo una recompensa media igual o superior a la politica de reglas.

Los resultados corresponden a una simulacion
academica. No representan pruebas en un robot fisico.

## Evidencias

- reports/figures/p2_cartpole_random.png
- reports/figures/p2_ppo_entrenamiento.png
- reports/figures/p2_09_recompensas.png
- reports/figures/p2_09_tasa_exito.png
- models/p2_ppo_robot.zip

# Practica 2 - Sesion 9

## Objetivo

Evaluar un agente PPO entrenado en CartPole-v1
y compararlo con un agente aleatorio.

## Metodologia

Se ejecutaron 10 episodios por agente
utilizando las mismas semillas iniciales.

PPO utiliza prediccion determinista.
El agente aleatorio selecciona acciones al azar.

## Resultados

| Metrica | PPO | Aleatorio |
|---|---:|---:|
| Promedio | 500.00 | 33.70 |
| Maxima | 500.00 | 93.00 |
| Minima | 500.00 | 10.00 |

Diferencia de recompensa promedio: 466.30

## Conclusion

PPO obtuvo una recompensa promedio superior a la del agente aleatorio.

Estos resultados corresponden a diez episodios
de simulacion. No garantizan el mismo rendimiento
con otras semillas o condiciones.

# Solucion de la ecuacion de Schrödinger con deep learning

En este repositorio, buscaremos las soluciones de la ecuación de Schrödinger para múltiples sistemas. Para ello, utilizaremos modelos de Deep Learning para cada sistema.

Comenzaremos por encontrar los estados estacionarios de la ecuación de Schrödinger independiente del tiempo: $x^1$
\begin{aligned}
\dot{x} & = \sigma(y-x) \\
\dot{y} & = \rho x - y - xz \\
\dot{z} & = -\beta z + xy
\end{aligned}

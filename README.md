# Solucion de la ecuacion de Schrödinger con deep learning

En este repositorio, buscaremos las soluciones de la ecuación de Schrödinger para múltiples sistemas. Para ello, utilizaremos modelos de Deep Learning para cada sistema.

Comenzaremos por encontrar los estados estacionarios de la ecuación de Schrödinger independiente del tiempo:

$$ \Big( - \frac{\hbar^2}{2m} \frac{\partial^2}{\partial x^2} + V(x) \Big) \phi_n(x) = E_n \phi_n(x) $$

Donde $\hbar$ es la constante de Plank reducida, $m$ es la masa de la partícula, $V(x)$ es el potencial bajo el cual evoluciona la partícula. $\phi_n(x)$ es ele $n$-th estado estacionario del sistema cuántico, con energía $E_n$. El valor $n$ puede ser discreto ($n \in \mathbb{Z}$) o continuo ($n \in \mathbb{R}$).

Las funciones de onda estacionaria $\phi_n(x)$ tener una evolución trivial en el tiempo:

$ \phi_n(x,t) = \phi_n(x) e^{-i \frac{E_n}{\hbar}t} $

Forman una base del espacio de Hilbert del hamiltoniano del sistema.

$$
\hat{H} = \frac{\hat{p}^2}{2m} + V(\hat{x})
$$

Por tanto, la evolución de un estado arbitrario del sistema $\psi(x,t)$ será dado por:

$$
\psi(x,t) = \sum_k c_k \phi_k(x) e^{-i \frac{E_k}{\hbar}t} 
$$

Donde $c_k$ son coeficientes lineales y la suma de $k$ puede ser discreta o continua (una integral). El módulo cuadrado de los coeficientes $|c_k|^2$ representa la probabilidad de estar en el $k$-th estado exitado. 

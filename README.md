# Solucion de la ecuacion de Schrödinger con deep learning

En este repositorio, buscaremos las soluciones de la ecuación de Schrödinger para múltiples sistemas. Para ello, utilizaremos modelos de Deep Learning para cada sistema.

Comenzaremos por encontrar los estados estacionarios de la ecuación de Schrödinger independiente del tiempo: 

<div align="center"> <img src="https://render.githubusercontent.com/render/math?math=\Big( - \frac{\hbar^2}{2m} \frac{\partial^2}{\partial x^2} + V(x) \Big) \phi_n(x) = E_n \phi_n(x)">  </div>

Donde  <img src="https://render.githubusercontent.com/render/math?math=\hbar"> es la constante de Plank reducida, <img src="https://render.githubusercontent.com/render/math?math=m"> es la masa de la partícula, <img src="https://render.githubusercontent.com/render/math?math=V(x)"> es el potencial bajo el cual evoluciona la partícula. <img src="https://render.githubusercontent.com/render/math?math=\phi_n(x)"> es ele n-th estado estacionario del sistema cuántico, con energía <img src="https://render.githubusercontent.com/render/math?math=E_n">. El valor n puede ser discreto (<img src="https://render.githubusercontent.com/render/math?math=n \in \mathbb{Z}">) o continuo (<img src="https://render.githubusercontent.com/render/math?math=n \in \mathbb{R}">).

Las funciones de onda estacionaria <img src="https://render.githubusercontent.com/render/math?math=\phi_n(x)"> tener una evolución trivial en el tiempo:

<div align="center"> <img src="https://render.githubusercontent.com/render/math?math=\phi_n(x,t) = \phi_n(x) e^{-i \frac{E_n}{\hbar}t}"> </div>

Forman una base del espacio de Hilbert del hamiltoniano del sistema.
<div align="center"> <img src="https://render.githubusercontent.com/render/math?math=\hat{H} = \frac{\hat{p}^2}{2m} + V(\hat{x})"> </div>

Por tanto, la evolución de un estado arbitrario del sistema $\psi(x,t)$ será dado por:
<div align="center"> <img src="https://render.githubusercontent.com/render/math?math=\psi(x,t) = \sum_k c_k \phi_k(x) e^{-i \frac{E_k}{\hbar}t}"> </div>


Donde <img src="https://render.githubusercontent.com/render/math?math=c_k"> son coeficientes lineales y la suma de k puede ser discreta o continua (una integral). El módulo cuadrado de los coeficientes <img src="https://render.githubusercontent.com/render/math?math=|c_k|^2"> representa la probabilidad de estar en el k-th estado exitado. 

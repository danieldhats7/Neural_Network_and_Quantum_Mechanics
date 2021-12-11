import utils.DataGen as DataGen
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    arr = np.array([1, 2, 3, 4])
    potencial = DataGen.potencial_datos()

    E_min, a_min = potencial.Calcular_energia_alpha_minimo(arr)
    print('Energia minima = {}'.format(E_min))
    print('alphas optimos = {}'.format(a_min))

    n_ejemplos = 100
    alpha_min = np.array([-4.5, -0.65, 0.2, -0.01, 0])
    alpha_max = np.array([1.5, 0.65, 1, 0.01, 0.1])
    k = len(alpha_min)
    estado_n = 0
    N = 10

    r_alpha = np.random.random((n_ejemplos, k))
    n_alphas = r_alpha*(alpha_max - alpha_min)+ alpha_min # random alpha
    E_mins = np.zeros(n_ejemplos)
    a_mins = np.zeros((n_ejemplos, N))

    for i in range(n_ejemplos):
        E_min, a_min = potencial.Calcular_energia_alpha_minimo(n_alphas[i,:], estado_n)
        E_mins[i] = E_min
        a_mins[i,:] = a_min
    print('E shape = {}'.format(E_mins.shape))
    print('a shape = {}'.format(a_mins.shape))


    x_min = -5
    x_max = 5
    n_points = 100

    waves, x, phis = potencial.final_wavefunction(x_min, x_max, n_points, a_mins)

    

    plt.figure(figsize=(10,6))
    sns.set_style("darkgrid")
    for i in range(3):
        sns.lineplot(x = x, y = waves[i,:])
    plt.show()
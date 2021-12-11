import numpy as np
from scipy.special import factorial
from scipy import linalg as LA
import scipy.sparse as sps
from scipy.linalg import eigh
from scipy.special import eval_hermite
from scipy.signal import argrelextrema

class potencial_datos:
    
    def __init__(self, alpha_min=None, alpha_max=None, N=10):
        
        if len(alpha_min)!=len(alpha_max):
            print("Error. Inconsisten shapes")

        self.alpha_min = np.array(alpha_min)
        self.alpha_max = np.array(alpha_max)
        self.N = N # Length of H.O basis
        self.k = len(alpha_min) #Number of alphas for V(x)

    def I_nmr(self,n, m, r):
        if r<0 or n<0 or m<0:
            return 0
        if r==0:
            if n==m:
                return np.sqrt(np.pi)*2**n*factorial(n)
            else:
                return 0
        return 1./2*self.I_nmr(n+1,m,r-1) + n*self.I_nmr(n-1,m,r-1)

    def C_nm(self, n, m, alphas):
        An = 1./np.sqrt(np.sqrt(np.pi)*factorial(n)*2**n)
        Am = 1./np.sqrt(np.sqrt(np.pi)*factorial(m)*2**m)
        I1 = -1/2*self.I_nmr(n,m,2)
        I2 = 1/2*self.I_nmr(n,m,0)
        I3 = 2*m*self.I_nmr(n,m-1,1)
        I4 = -2*m*(m-1)*self.I_nmr(n, m-2,0)
        Iv = 0
        for i in range(len(alphas)):
            Iv+=alphas[i]*self.I_nmr(n,m,i)
        return An*Am*(I1+I2+I3+I4+Iv)

    def Calcular_energia_alpha_minimo(self, alphas, estado_n=0):
        N = self.N
        # Generando la matriz C_nm
        C = np.zeros((N,N))
        for n in range(N):
            for m in range(N):
                C[n,m] = self.C_nm(n,m,alphas)

        # Generando la matriz D
        D = np.zeros((N,N))
        for n in range(N):
            for m in range(n+1):
                D[n,m] = C[n,m] + C[m,n]
                D[m,n] = D[n,m]

        # Diagonalizando la matriz D
        vaps, veps = eigh(D)

        # Calcular <H> 
        Hs = np.zeros(N)
        for i in range(N):
            a = veps[:, i]
            for n in range(N):
                for m in range(N):
                    Hs[i]+=a[n]*a[m]*C[n,m]

        # 4. We choose the vector which minimizes <H>
        # If n_state!=0, we choose the vector with n_state-th lowest energy
        # as an approximation of the n_state excited state 
        sel = np.argsort(Hs)[estado_n] #np.argmin(Hs)
        a = veps[:, sel] # Final value of eigenvalues for state n_state
        E_a = Hs[sel] # Value of the energy
        return E_a, a

    def HO_wavefunction(self, n, xmin, xmax, n_points):
        x = np.arange(xmin, xmax, (xmax - xmin)/n_points)
        herm = eval_hermite(n, x) # H_n(x)
        exp = np.exp(- x**2/2) # Exponential term
        phi_n = exp*herm

        # Normalization
        h = (xmax - xmin)/n_points
        C = 1./np.sqrt(np.sum(phi_n*phi_n*h))
        phi_n = C*phi_n
        
        return phi_n

    def final_wavefunction(self, xmin, xmax, n_points, a_s):
        x = np.arange(xmin, xmax, (xmax - xmin)/n_points)
        n_samples, _ = a_s.shape
        # Construct matrix of phi_n
        phis = np.zeros((self.N, n_points))

        for i in range(self.N):
            phis[i,:] = self.HO_wavefunction(i, xmin, xmax, n_points)

        waves = np.zeros((n_samples, n_points))
        for i in range(n_samples):
            for j in range(n_points):
                waves[i,j] = np.dot(a_s[i,:],phis[:,j])
            # convention: To choose the phase we make the maximums be first
            w = waves[i,:]
            maxi = argrelextrema(w, np.greater)[0] #array con indices de maximos localles de w
            mini = argrelextrema(w, np.less)[0]    #array con indices de minimos localles de w
            idx2= np.abs(w[maxi])>5e-2 
            maxi = maxi[idx2] #idices con valores absolutos maximos de la funcion wave > 0.05
            idx2= np.abs(w[mini])>5e-2 
            mini = mini[idx2] #idices con valores absolutos minimos de la funcion wave > 0.05
            if len(maxi)==0 and len(mini)>0:
                waves[i,:] = -waves[i,:]
            elif len(mini)>0 and len(maxi)>0 and mini[0]<maxi[0]:
                waves[i,:] = -waves[i,:]
        return waves, x, phis

    def evaluar_potencial(self, xmin, xmax, n_points, n_alphas):
        x = np.arange(xmin, xmax, (xmax - xmin)/n_points)
        n_samples, k = n_alphas.shape
        V = np.zeros((n_samples, n_points))
        x_mat = (x**np.arange(k)[:,None])# Matrix of powers of x: x^0, x^1, x^2, ..., x^N (in every row)
        V = np.zeros((n_samples, n_points))# V(x) in each row different alpha
        for i in range(n_samples):
            for j in range(n_points):
                V[i,j] = np.dot(n_alphas[i,:],x_mat[:,j])
        return V, x

    def generate_data(self, n_samples, alpha=np.array([None]), n_state=0, display=100):
        '''
        Generates samples of potentials  with random coefficients and finds the n_state excited state for them
        Args:
            n_samples (int): Number of samples of potentials (alphas)
            alpha (np.array): Values of alpha. If you want to generate them randomly, don't provide anything
            n_state (int): Number of excited state (default n_state=0, ground state)
            display (int): Display step
        Returns:
            E (np.array): size n_samples. Ground energy for each V
            a (np.array): size n_samples x N. Coefficients in the H.O basis for each V
            alpha (np.array): size n_samples x k. Coefficients of the potentials V(x)
        '''
        data = np.zeros((n_samples, self.N))

        # Generate random value of alphas
        if (alpha==None).any():
            print("Random alphas")
            r_alpha = np.random.random((n_samples, self.k)) # Values between 0 and 1
            alpha = r_alpha*(self.alpha_max - self.alpha_min)+ self.alpha_min # random alpha
        
        # Prepare vectors of energies and coefficients
        E = np.zeros(n_samples)
        a = np.zeros((n_samples, self.N))
        # Find ground state for each sample
        for i in range(n_samples):
            E_new, a_new = self.Calcular_energia_alpha_minimo(alpha[i,:], n_state)
            if i%display==0:
                print("\rGenerating data: {}/{}".format(i,n_samples), end='')
            E[i] = E_new
            a[i,:] = a_new
        return E, a, alpha  
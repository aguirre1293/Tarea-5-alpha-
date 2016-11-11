import numpy as np
import matplotlib.pyplot as plt

G = 6.67408*10**(-11)
AU = 1.4960*10**(11)
yr = 3.154*10**(7)

x_obs = np.array([0.324, -0.702, -0.982, 1.104, 3.266, -9.219, 19.931, 24.323])*AU
y_obs = np.array([0.091, -0.169, -0.191, -0.826, -3.888, 1.788, 2.555, -17.606])*AU
z_obs = np.array([-0.022, 0.038, -0.000, -0.045, -0.057, 0.336, -0.267, 0.198])*AU

vx_obs = np.array([-4.628, 1.725, 1.127, 3.260, 2.076, -0.497, 0.172, 0.664])*AU/yr
vy_obs = np.array([10.390, -7.205, -6.188, 4.524, 1.904, -2.005, 1.357, 0.935])*AU/yr
vz_obs = np.array([1.274, -0.199, 0.000, 0.015, -0.0543, 0.0547, 0.003, -0.035])*AU/yr

def likelihood(v_model):
	v_obs = np.zeros(len(vx_obs))
	for i in range(0,len(vx_obs)):
		v_obs[i] = np.log10((vx_obs[i]**2+vy_obs[i]**2+vz_obs[i]**2)**0.5)
	return 1./(2.*10**(-18))*sum(-(v_obs-v_model)**2)

def my_model(a,M):
	m = (a-1.)/2.
	b = 1./2.*(np.log10(G)+M)
	result = np.zeros(len(x_obs))
	for i in range(len(x_obs)):
		r = (x_obs[i]**2+y_obs[i]**2+z_obs[i]**2)**(0.5)
		result[i] = -m*np.log10(r)+b
	return result

n_points = 10**4

a_walk = np.zeros(n_points)
M_walk = np.zeros(n_points)
l_walk = np.zeros(n_points)

a_walk[0] = 2
M_walk[0] = 30 #M es el log de la masa del Sol

v_init = my_model(a_walk[0],M_walk[0])
l_init = likelihood(v_init)

for i in range(0,n_points-1):
	a_prime = np.random.normal(a_walk[i],0.1)
	M_prime = np.random.normal(M_walk[i],0.1)
	
	v_init = my_model(a_walk[i],M_walk[i])
	v_prime = my_model(a_prime,M_prime)

	l_init = likelihood(v_init)
	l_prime = likelihood(v_prime)

	comp = np.exp(-(l_init-l_prime))
	if comp >= 1.0:
		a_walk[i+1] = a_prime
		M_walk[i+1] = M_prime
		l_walk[i+1] = l_prime
	else:
		a_walk[i+1] = a_walk[i]
		M_walk[i+1] = M_walk[i]
		l_walk[i+1] = l_init

plt.scatter(a_walk,M_walk)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

sigma = 0.1
v = 5

x_obs = [3.,3.,4.,4.,5.,5.]
y_obs = [15.,16.,15.,16.,15.,16]
t_obs = [3.12, 3.26, 2.98, 3.12, 2.84, 2.98]

def likelihood(x_guess,y_guess):
	n = len(t_obs)
	result = 0
	for i in range(n):
		dist = ((x_obs[i]-x_guess)**2+(y_obs[i]-y_guess)**2)**0.5
		time = v/dist
		result += -(t_obs[i]-time)**2/(2*sigma**2)
	return result

n_points = 10**4

x_walk = np.zeros(n_points)
y_walk = np.zeros(n_points)
l_walk = np.zeros(n_points)

x_walk[0] = 10
y_walk[0] = 10.

l_walk[0] = likelihood(x_walk[0],y_walk[0])

for i in range(0,n_points-1):
	x_prime = np.random.normal(x_walk[i],sigma)
	y_prime = np.random.normal(y_walk[i],sigma)
	
	l_prime = likelihood(x_prime,y_prime)
	l_init = likelihood(x_walk[i],y_walk[i])

	alpha = np.exp(-(-l_prime+l_init))
	if alpha>= 1.0:
		x_walk[i+1] = x_prime
		y_walk[i+1] = y_prime
		l_walk[i+1] = l_prime
	else:
		x_walk[i+1] = x_walk[i]
		y_walk[i+1] = y_walk[i]
		l_walk[i+1] = l_init

plt.scatter(x_walk[100:],y_walk[100:])
plt.show()





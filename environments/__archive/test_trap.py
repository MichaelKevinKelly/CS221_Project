from mcts import MCTS
from nodes import *
from motion_domain_1d import dynamics, random_act_generator, terminal_estimator
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(15)
mcts = MCTS(dynamics,random_act_generator,terminal_estimator=terminal_estimator,rng=rng)
# state = np.zeros(2)
state = 0.

def plot(state):
	obst=plt.Circle((0.5, 0.5), 0.05, color='r')
	goal=plt.Circle((1., 1.), 0.01, color='g')
	agent=plt.Circle(tuple(state/30.), 0.01, color='b')
	plt.gcf().gca().add_artist(obst)
	plt.gcf().gca().add_artist(goal)
	plt.gcf().gca().add_artist(agent)
	plt.show()

steps = 60
r_sum = 0.
for i in range(steps):
	# plot(state)
	action = mcts.plan(state)
	state,r = dynamics(state,action,rng)
	print(state)
	print(r)
	r_sum+=r
print('Average reward: {}'.format(r_sum/steps))

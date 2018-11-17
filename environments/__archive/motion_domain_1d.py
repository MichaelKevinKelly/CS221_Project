import numpy as np

dt = 1.
sigma = 0.05
# goal = np.array([0.,5.])
# obstacle = np.array([2.5,2.5])
# obstacle_r = 0.5
num_steps = 5
R = 1. #0.25

def dynamics(state,action,rng):
	# act = np.clip(action,-1.,1.) ## Saturate
	nxt_state = state + action*R
	nxt_state += np.random.randn()*0.01
	# nxt_state += rng.multivariate_normal(np.zeros(2),sigma*np.eye(2))
	# d_goal = np.linalg.norm(goal-nxt_state)
	# d_obst = np.linalg.norm(obstacle-nxt_state)
	# r = 1/float(d_goal) if d_obst>obstacle_r else -5
	r = 0
	if nxt_state<1.:
		r = 70
	elif nxt_state>5.:
		r = 500
	return nxt_state, r
	
def random_act_generator(state,rng):
	return rng.rand()-0.1

def terminal_estimator(leaf,discount,rng):
	return leaf.state*0.01#[0]
	# state = leaf.state
	# v = 0
	# for i in range(num_steps):
	# 	act = random_act_generator(None,rng)
	# 	state,r = dynamics(state,act,rng)
	# 	v = r + discount * v
	# return v

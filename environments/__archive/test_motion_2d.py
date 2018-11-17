import numpy as np
import matplotlib.pyplot as plt
from motion_domain_2d import motion_domain_2d
from planner import Planner

env = MotionDomain2d()
params_path = 'mcts_parameters.yaml'
planner = Planner(env,params_path)
rng = Planner.rmcts.rng

state, action = None, 0
r_cum = 0
while planner.risk_handler.get_residual_steps()>0 and action is not None:
	# env.plot(curr_state)
	action = Planner.plan(state)
	state,r = dynamics(state,action,rng)
	r_cum += r
result = 'Failure' if action is None else 'Success'
print(result + ', total reward = {}'.format(r_sum))
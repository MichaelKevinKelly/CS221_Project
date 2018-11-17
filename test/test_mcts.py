from planner.ar_mcts_v1 import RAMCTS
from environments.test_mcts_env import BasicMCTSTestEnv
import numpy as np
import yaml

params_path = 'mcts_test_parameters.yaml'
params = yaml.load(open(params_path))['parameters']
test_env = BasicMCTSTestEnv()

mcts = RAMCTS(test_env,
		params['exploration_type'],
		niter = params['niter'],
		c = params['c'],
		lmbda = params['lmbda'],
		rng = np.random.RandomState(params['random_seed']),
		k_s = params['k_s'],
		k_a = params['k_a'],
		alpha_s = params['alpha_s'],
		alpha_a = params['alpha_a'],
		p = params['p'],
		discount = params['discount'],
		return_most_simulated = params['return_most_simulated'],
		risk_aware = params['risk_aware'],
		single_step_risk_aware = params['single_step_risk_aware'],
		conservative_risk_prop = params['conservative_risk_prop'])

init_state = 0.
risk_bound = 1.
horizon = 30

state = init_state
q = 0
reward = 0
for i in range(horizon):
	print('State at iteration {}: {}'.format(i,state))
	print('Reward: {}'.format(reward))
	act,risk = mcts.plan(state,risk_bound,horizon)
	state,reward = test_env.dynamics(state,act,mcts.rng)
	q += reward
print('Total reward: {}'.format(q))
print('Avg reward: {}'.format(q/horizon))
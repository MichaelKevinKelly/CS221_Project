from planner.ar_mcts_v1 import RAMCTS
from planner.risk_handler import RiskHandler
import numpy as np

def RecedingHorizonPlanner(object):

	def __init__(self,env,params):
		# params = yaml.load(open(params_path))['parameters']
		self.env = env
		self.risk_handler = RiskHandler(params['total_risk_bound'],params['total_horizon'],params['mcts_horizon'])
		self.mcts = generate_mcts(env,params)

	def plan(self,state):
		action, risk = None, None
		while action is None:
			curr_mcts_risk_bound, curr_mcts_horizon, _break = self.risk_handler.get_mcts_risk_allocation()
			if _break: break
			action, risk = self.mcts.plan(state, curr_mcts_risk_bound, curr_mcts_horizon)
		return action, risk

	def update(self,risk):
		self.risk_handler.step(risk)

	def generate_mcts(self,env,params):
		mcts = RAMCTS(env,
			params['exploration_type'],
			niter = params['niter'],
			horizon = params['mcts_horizon'],
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
			conservative_risk_prop = params['conservative_risk_prop']
		)
		return mcts

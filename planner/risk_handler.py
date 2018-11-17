import numpy as np

class RiskHandler(object):

	def __init__(self,risk_bound,horizon,mcts_horizon):
		self.total_risk_bound = risk_bound
		self.total_horizon = horizon
		self.mcts_horizon = mcts_horizon
		self.residual_risk = risk_bound
		self.cumulative_risk = 0
		self.step = 0
		self.i = 0

	def step(self,step_risk):
		self.cumulative_risk += step_risk
		self.step += 1
		self.residual_risk = self.total_risk_bound - self.cumulative_risk
		self.residual_steps = self.total_horizon - self.step
		self.num_windows = int(self.residual_steps / self.mcts_horizon)
		self.i = 0

	def get_mcts_risk_allocation(self):
		_break = self.i>5
		if self.num_windows>0:
			curr_mcts_horizon = self.mcts_horizon
			p_curr = curr_mcts_horizon/self.residual_steps
			p_remaining = 1. - p_curr
			p_curr_ext = p_curr + (self.i/5.)*(p_remaining)
			curr_risk_bound = p_curr_ext * self.residual_risk
		else:
			_break = self.i>0
			curr_risk_bound = self.residual_risk
			curr_mcts_horizon = residual_steps
		self.i += 1
		return curr_mcts_risk_bound, curr_mcts_horizon, _break
import numpy as np

class DPWNode(object):
	def __init__(self,parent,k,alpha,init_n=0,init_q=0,risk=0):
		self.parent = parent
		self.children = []
		self._k = k
		self._alpha = alpha
		self.N = init_n
		self.risk = risk

	@property
	def max_children(self):
		return int(self._k * min(np.power(self._N,self._alpha),1))

class DpwActionNode(DPWNode):
	def __init__(self,act,parent,k,alpha,init_n=0,init_q=0):
		DPWNode.__init__(self,parent=parent,k=k,alpha=alpha,init_n=init_n,init_q=init_q)
		self.act = act
		self.Q = init_q
		self.unnormalized_risk = 0.
		self.violates_bound = False

	def update_risk_bound_violation(self,bound):
		self.risk = self.unnormalized_risk / self.N
		self.violates_bound = self.risk > bound

class DpwStateNode(DPWNode):
	def __init__(self,state,parent,reward,k,alpha,risk_bound=None,init_n=1,risk=0.):
		DPWNode.__init__(self,parent,k,alpha,init_n=init_n,risk=risk)
		self.state = state
		self.reward = reward
		self.sim_val = None
		self.prev_risk = 0.
		self.child_risk_status = []
		self.risk_bound = risk_bound
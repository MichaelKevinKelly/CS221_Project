import numpy as np

## constants = {k: None, alpha: None, k_a:-, alpha_a:-, k_s:-, alpha_s:-, dmax:-, exp_type:-, p:-}

class DpwNode(object):
	def __init__(self,parent,depth,constants,k,a,init_n=0):
		## Tree data
		self.parent = parent
		self.children = []
		self.exploration_type = constants['exploration_type']
	
		## Node data
		self.N = init_n
		
		## Set up node expansion / exploration
		if self.exploration_type=='ucb':
			self.k = k
			self.alpha = a
			depth = None
			self.allow_new_func = lambda: len(self.children)<=np.floor(self.k * self.N**self.alpha)
			self.max_children_ka = lambda: self.k * self.N**self.alpha
		elif self.exploration_type=='polynomial' or self.exploration_type=='polynomial_heuristic':
			if self.exploration_type=='polynomial':
				depth_diff = constants['dmax'] - depth
				if abs(np.rint(depth)-depth)<1e-5: ## If depth is integer (in state node)
					self.alpha = (10.*depth_diff-3.)**-1
				else: ## If depth is not integer (in action node)
					if depth_diff>=1.5:
						self.alpha = 3./(10.*depth_diff-3.)
					elif abs(abs(depth_diff)-0.5)<1e-5:
						self.alpha = 1.
					else:
						assert(False)
				self.e_d = (1./(2.*constants['p']))*(1.-3./(10*depth_diff))
			else:
				self.k = k
				self.alpha = a
				self.e_d = constants['e_d']
			self.allow_new_func = lambda: np.floor(self.N**self.alpha)>np.floor((self.N-1)**self.alpha)
		else:
			raise NotImplementedError

	@property
	def allow_new_node(self):
		return self.allow_new_func()

	@property
	def get_max_children_ka(self):
		return self.max_children_ka()
		
class DpwActionNode(DpwNode):
	def __init__(self,act,parent,depth,constants,init_n=0,init_q=1,risk=0.):
		assert(abs(np.modf(depth)[0]-0.5)<1e-5)
		DpwNode.__init__(self,parent,depth,constants,constants['k_a'],constants['alpha_a'],
			init_n=init_n)
		self.act = act
		self.Q = init_q
		self.risk = risk
		self.N_immediate_violations = 0
		self.e_d = None

	@property
	def immediate_risk(self):
		return float(self.N_immediate_violations)/self.N

class DpwStateNode(DpwNode):
	def __init__(self,state,reward,parent,depth,constants,violates_constraint,init_n=1):
		assert(abs(np.rint(depth)-depth)<1e-5)
		DpwNode.__init__(self,parent,depth,constants,constants['k_s'],constants['alpha_s'],init_n=init_n)
		self.state = state
		self.reward = reward
		self.violates_constraint = violates_constraint

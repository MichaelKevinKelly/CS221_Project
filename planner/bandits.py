import numpy as np

def get_nfunc(exploration_type):
		if exploration_type=='ucb':
			get_ncount = lambda s_node: np.log(float(s_node.N))
		elif exploration_type=='polynomial' or exploration_type=='polynomial_heuristic':
			get_ncount = lambda s_node: float(s_node.N)**s_node.e_d
		else:
			raise NotImplementedError
		return get_ncount

def bandit_rule_generator(exploration_type, params):
	
	## Default c,lmda for polynomial exploration = 1,1
	c = params['c']
	lmbda = params['lmbda']
	risk_aware = params['risk_aware']

	get_ncount = get_nfunc(exploration_type)

	def select_action(state_node,risk_bound):
		assert(len(state_node.children)>0)
		parent_ncount = get_ncount(state_node)
		vals = []
		for a in state_node.children:
			eps = np.sqrt(parent_ncount/(a.N+1e-10))
			val = a.Q + c * eps ## Upper bound on Q-function
			if risk_aware:
				r_hat_lb = a.risk - lmbda * eps ## Lower bound on risk
				val *= float(r_hat_lb<=risk_bound)
			vals.append(val)
		return state_node.children[np.argmax(vals)]

	return select_action
import numpy as np
from nodes import DpwStateNode, DpwActionNode

class R_MCTS(object):

	def __init__(self,
			dynamics,
			random_act_generator,
			constraint_checker,
			risk_bound,
			horizon,
			terminal_estimator = lambda x,y,z: 0,
			niter = 5000,
			max_depth = 30,
			c = 3.0,
			rng = np.random.RandomState(1),
			init_q = 0.0,
			init_n = 0,
			k_s = 15.,
			k_a = 15.,
			alpha_s = 0.5,
			alpha_a = 0.5,
			discount = 1.,
		):

		self.dynamics = dynamics
		self.constraint_checker = constraint_checker
		self.terminal_estimator = terminal_estimator
		self.random_act_generator = random_act_generator
		
		self.risk_bound = risk_bound
		self.residual_risk_alloc = risk_bound
		self.step = 0
		self.horizon = horizon
		self.niter = niter
		self.max_depth = max_depth
		self.c = c
		self.discount = discount
		self.rng = rng
		self.init_q = init_q
		self.init_n = init_n
		self.k_s = k_s
		self.k_a = k_a
		self.alpha_s = alpha_s
		self.alpha_a = alpha_a
		
	def plan(self,init_state):
		root = DpwStateNode(init_state,None,0,self.k_s,self.alpha_s,init_n=self.init_n)
		d = 0
		for it in range(self.niter):
			leaf, _d = self.simulate(root)
			val = self.terminal_estimator(leaf,self.discount,self.rng)
			self.backup(leaf,val)
			d = max(d,_d)
		act = self.select_action(root,0).act
		print('Max depth: {}'.format(d))
		self.step += 1
		##TODO: 
		return act

	## Determine in-tree simulation
	def simulate(self,root):
		root.N += 1
		curr_node, d = root, 0
		_break = False
		while d<self.max_depth and not _break:
			d += 1
			act_node = self.action_pw(curr_node)
			curr_node, _break = self.state_pw(act_node,d)
		return curr_node, d

	## Action Progressive Widening
	def action_pw(self,state_node,ucb=True):
		## Add a new action if allowed
		if len(state_node.children)<state_node.max_children:	
			act = self.random_act_generator(state_node.state,self.rng)
			new_act_node = DpwActionNode(act,state_node,self.k_a,self.alpha_a,
				init_q=self.init_q,init_n=self.init_n)
			state_node.children.append(new_act_node)
		else:
			# print('Maxed out actions')
			pass
		## Choose from among existing actions acording to UCB1
		act_node = self.select_action(state_node, self.c if ucb else 0)
		act_node.N += 1
		return act_node

	## State Progressive Widening
	def state_pw(self,act_node,depth):
		## Sample a new successor state if allowed
		init_state_node = act_node.parent
		_break = False
		if len(act_node.children)<act_node.max_children:
			state, r = self.dynamics(init_state_node.state,act_node.act,self.rng)
			state_risk = self.get_state_risk(state,depth)
			state_node = DpwStateNode(state,act_node,r,self.k_s,self.alpha_s,
				init_n=self.init_n,risk=state_risk)
			act_node.children.append(state_node)
			_break = True
		else:
			# print('Maxed out states')
			state_node = self.sample_successor_state(act_node)
			_break = abs(state_node.risk-1.)<1e-6 ## Don't recurse on states that violate constraints
		state_node.N += 1
		return state_node, _break

	## Get risk at a leaf state
	def get_state_risk(self,state,depth):
		violation = self.constraint_checker(state,depth)
		if violation:
			return 1.
		else:
			p_residual = float(self.horizon-depth)/self.horizon
			risk_heuristic = p_residual * self.risk_bound
			return risk_heuristic

	## Implements UCB1
	def select_action(self,state_node,c):
		assert(len(state_node.children)>0)
		log_N = np.log(float(state_node.N))
		vals = [a.Q + c*np.sqrt(log_N/(a.N+1e-5)) for a in state_node.children]
		return state_node.children[np.argmax(vals)]

	## Samples successor state from (s,a) based on visitation frequency
	def sample_successor_state(self,act_node):
		N = float(act_node.N-1)
		state_nodes = act_node.children
		probs = np.array([state_node.N/N for state_node in state_nodes])
		i = self.rng.choice(len(state_nodes),p=probs)
		return state_nodes[i]

	## Back up values up tree
	def backup(self,child_state_node,val):
		action_node = child_state_node.parent
		q = val
		i = 0
		while action_node is not None:
			q = child_state_node.r + self.discount * q
			action_node.Q += (q - action_node.Q) / action_node.N ##s.t. risk bound?? --> no, we want value calculated independently
			self.update_risks(action_node,child_state_node,i)
			i+=1
			child_state_node = action_node.parent
			action_node = child_state_node.parent

	## Update a state node's risk estimate
	## Returns the previous risk estimate
	def update_risks(self,action_node,child_state_node,height):
		r_child_prev = child_state_node.prev_risk
		r_child = child_state_node.risk
		r_a = action_node.risk
		pr_as = float(child_state_node.N)/action_node.N
		parent_state_node = action_node.parent
		if height==0:
			## Weighted average update
			action_node.risk = r_a_curr + (r_child - r_a_curr) ## THis should be a *weighted* average
			##TODO: need to update transition probabilities across *all* subsequent states
		else:
			## Remove previous risk contribution of child state
			risk_excl = (N * r_a - r_c_prev) / (N - 1)

			## Add new risk contribution
			## Cache new risk contribution
		## Update parent state node risk - max_a V() s.t. risk < bound


		r_a = action_node.risk
		r_c = child_state_node.risk
		N_a = action_node.N
			

			if i==0:
				action_node.risk = r_a + (r_c - r_a) / N
			else:
				## Update state 
				

				## Need to subtract out previous risk value (i.e. calculate r_excl)
				r_excl = (N * r_a - r_c_prev) / (N - 1)
				action_node.risk = r_excl + (r_c - r_excl) / N
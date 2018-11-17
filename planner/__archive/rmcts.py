import numpy as np
from nodes_v0 import DpwStateNode, DpwActionNode

class RMCTS(object):

	def __init__(self,
			env,
			bandit_rule,
			risk_heuristic=None,
			niter = 5000,
			max_depth = 30,
			c = 3.0,
			rng = np.random.RandomState(1),
			init_q = 0.0,
			k_s = 15.,
			k_a = 15.,
			alpha_s = 0.5,
			alpha_a = 0.5,
			discount = 1.,
		):

		## Problem-specific parameters
		self.dynamics = env.dynamics
		self.constraint_checker = env.constraint_checker
		self.terminal_estimator = env.terminal_estimator
		self.random_act_generator = env.random_act_generator
		self.risk_heuristic = risk_heuristic

		## MCTS parameters
		self.bandit_rule = bandit_rule ## Determines action selection scheme for tree policy
		self.niter = niter
		self.max_depth = max_depth
		self.c = c
		self.discount = discount
		self.rng = rng
		self.init_q = init_q
		self.k_s = k_s
		self.k_a = k_a
		self.alpha_s = alpha_s
		self.alpha_a = alpha_a
		
	def plan(self,init_state,risk_bound):
		root = DpwStateNode(init_state,None,0,self.k_s,self.alpha_s,risk_bound=risk_bound,init_n=0)
		d = 0
		for it in range(self.niter):
			leaf, _d = self.simulate(root)
			d = max(d,_d)
			val = self.terminal_estimator(leaf,self.discount,self.rng)
			self.backup(leaf,val)
		act = self.select_action(root,exploration=False)
		print('Max depth: {}'.format(d))
		return None if act is None else act.act

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
				init_q=self.init_q,init_n=0)
			state_node.children.append(new_act_node)
		else:
			# print('Maxed out actions')
			pass
		## Choose from among existing actions acording to bandit rule
		act_node = self.select_action(state_node,exploration=True)
		act_node.N += 1
		return act_node

	## State Progressive Widening
	def state_pw(self,act_node,depth):
		## Sample a new successor state if allowed
		init_state_node = act_node.parent
		new = False
		if len(act_node.children)<act_node.max_children:
			state, r = self.dynamics(init_state_node.state,act_node.act,self.rng)
			state_risk = self.get_state_risk(state,depth)
			state_node = DpwStateNode(state,act_node,r,self.k_s,self.alpha_s,init_n=1,risk=state_risk)
			act_node.children.append(state_node)
			_break = True
		else:
			# print('Maxed out states')
			state_node = self.sample_successor_state(act_node)
		violation = self.constraint_checker(state_node.state)
		if violation:
			act_node.N_immediate_violations += 1
			_break = True ## Don't recurse on states that violate constraints
		state_node.N += 1
		return state_node, _break

	def select_action(self,state_node,exploration):
		assert(len(state_node.children)>0)
		if exploration:
			log_parent_cnt = np.log(state_node.N)
			vals = [self.bandit_rule(a,log_parent_cnt,self.risk_bound) for a in state_node.children]
		else:
			vals = [a.Q for a in state_node.children if not a.violates_bound]
		ret = None if len(vals)==0 else state_node.children[np.argmax(vals)]
		return ret

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
		while action_node is not None:
			q = child_state_node.r + self.discount * q
			action_node.Q += (q - action_node.Q) / action_node.N
			self.update_risks(action_node,child_state_node)
			child_state_node = action_node.parent
			action_node = child_state_node.parent

	def get_state_risk(self,state,depth):
		violation = self.constraint_checker(state)
		return self.risk_heuristic(state,depth) if not violation else 1.

	def update_risks(self,action_node,child_state_node):
		## Update the state risk
		state_risk_prev = child_state_node.risk
		
		state_node_children = child_state_node.children
		if len(state_node_children)>0: ## not a leaf state
			
			best_valid_act, best_valid_act_val = None, -np.inf
			min_risk_act, min_risk_act_risk = None, np.inf
			
			for a in state_node_children:
				violates_bound, risk, val = a.violates_bound, a.risk, a.Q
				if risk < min_risk_act_risk:
					min_risk_act, min_risk_act_risk = a, risk
				if not violates_bound and val > best_valid_act_val:
					best_valid_act, best_valid_act_val = a, val
			
			assert(min_risk_act_risk<=1.)
			
			if best_valid_act is not None:
				state_risk = best_valid_act.risk
			else:
				state_risk =  min_risk_act_risk
		
		else: ## a leaf state (new state or risk(state)=1 or max depth reached)
			state_risk = state_risk_prev
		child_state_node.risk = state_risk

		## Update the action risk
		state_n = child_state_node.N
		action_node.unnormalized_risk += (state_n * state_risk - (state_n-1) * prev_state_risk) ## Second term should be zero for new state nodes
		action_node.update_risk_bound_violation(self.risk_bound)
		child_state_node.prev_risk = state_risk

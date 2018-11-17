import numpy as np
from nodes import DpwStateNode, DpwActionNode
from bandits import bandit_rule_generator

class RAMCTS(object):

	def __init__(self,
			env,
			exploration_type,
			niter = 5000,
			horizon = 30,
			c = 3.0,
			lmda = 3.0,
			rng = np.random.RandomState(1),
			k_s = 15.,
			k_a = 15.,
			alpha_s = 0.5,
			alpha_a = 0.5,
			discount = 1.,
			init_q = lambda state,act=None: 0,
			return_most_simulated = False,
			risk_aware = True,
			single_step_risk_aware = True
		):

		## Problem-Specific Parameters
		self.dynamics = env.dynamics
		self.rollout_policy = env.rollout_policy
		self.random_act_generator = env.random_act_generator
		self.init_q = env.init_q if hasattr(env, 'init_q') else init_q
		
		## General MCTS Parameters
		self.niter = niter
		self.horizon = horizon
		self.discount = discount
		self.return_most_simulated = return_most_simulated
		self.c = c
		self.exploration_type = exploration_type
		self.node_constants = {'k':None, 'alpha':None, 
			'k_s':k_s, 'k_a':k_a, 'alpha_s':alpha_s, 'alpha_a':alpha_a,
			'dmax': horizon, 'exploration_type': exploration_type, 'p': p}
		self.select_action = bandit_rule_generator(exploration_type,{'c':c,'lmbda':lmbda, 'use_risk':risk_aware})
		self.single_step_risk_aware = single_step_risk_aware
		
		## DPW Parameters
		self.k_s = k_s
		self.k_a = k_a
		self.alpha_s = alpha_s
		self.alpha_a = alpha_a

		## RAMCTS Parameter
		self.lmda = lmda

		## Random Number Generator
		self.rng = rng
	
	## Plan from init_state to a horizon of self.horizon while respecting risk_bound
	def plan(self,init_state,risk_bound):
		root = DpwStateNode(init_state,0,None,self.k_s,self.alpha_s,
			init_q=self.init_q(init_state),risk_bound=risk_bound)
		d = 0
		for it in range(self.niter):
			leaf, curr_depth = self.simulate(root)
			d = max(d,curr_depth)
			val,risk = self.terminal_estimator(leaf,curr_depth)
			self.backup(leaf,val,risk)
		self.prune_tree(root,risk_bound,conservative=True)
		act = self.select_single_action(root)
		print('Max depth: {}'.format(d))
		return act.act, act.immediate_risk

	## Choose single action to execute from root state
	def select_single_action(self,root):
		actions = root.children
		if single_step_risk_aware:
			risk_bound = root.risk_bound
			actions = [a for a in actions if a.risk<=risk_bound]
		if self.return_most_simulated:
			action_vals = [a.N for a in actions]
		else:
			action_vals = [a.Q for a in actions]
		return actions[np.argmax(action_vals)]

	## Performs rollout of tree policy
	def simulate(self,root):
		root.N += 1
		curr_node, d = root, 0
		new_state = False
		while d<self.horizon and not _break:
			act_node = self.action_pw(curr_node)
			curr_node, _break = self.state_pw(act_node)
			d += 1
		return curr_node, d

	## Implements action progressive widening (see https://hal.archives-ouvertes.fr/hal-00542673v1/document)
	def action_pw(self,state_node,ucb=True):
		## Add a new action if allowed
		if len(state_node.children)<state_node.max_children:	
			act = self.random_act_generator(state_node.state,self.rng)
			new_act_node = DpwActionNode(act,state_node,self.k_a,self.alpha_a,
				init_q=self.init_q,init_n=1,risk_bound=state_node.risk_bound)
			state_node.children.append(new_act_node)
		else:
			# print('Maxed out actions')
			pass
		## Choose from among existing actions acording to risk-ucb
		act_node = self.select_action_rucb(state_node)
		act_node.N += 1
		return act_node

	## Implements state progressive widening (see https://hal.archives-ouvertes.fr/hal-00542673v1/document)
	def state_pw(self,act_node):
		## Sample a new successor state if allowed
		init_state_node = act_node.parent
		_break = False
		if len(act_node.children)<act_node.max_children:
			state, r = self.dynamics(init_state_node.state,act_node.act,self.rng)
			state_node = DpwStateNode(state,act_node,r,self.k_s,self.alpha_s,init_n=1,risk=0.)
			act_node.children.append(state_node)
			_break = True
		else:
			# print('Maxed out states')
			state_node = self.sample_successor_state(act_node)
		state_node.N += 1
		self.propagate_risk_bound(act_node,state_node)
		_break = _break or self.violates_constraint(state_node.state)
		return state_node, _break

	## Propagates risk bounds from an action node to one of its children
	def propagate_risk_bound(self,act_node,state_node):
		## Accumulate info on parent action and child state
		assert(state_node in act_node.children)
		children = action_node.children
		N,risk_bound = float(action_node.N),action_node.risk_bound
		curr_p,curr_r = state_node.N/N, state_node.r

		## Accumulate info on neighbors
		probs = [s.N/N for s in children]
		risks = [s.R for s in children]
		sum_risk_excl = np.sum([p*r for p,r in zip(probs,risks)])-curr_p*curr_r

		## Propagate new bound to child state
		new_bound = (1./curr_p)*(risk_bound-sum_risks_excl)
		state_node.risk_bound = new_bound
		
	## Implements risk UCB
	def select_action_rucb(self,state_node):
		assert(len(state_node.children)>0)
		N, children = float(state_node.N), state_node.children
		risk_bound = state_node.risk_bound
		c, lmda = self.c, self.lmda
		vals = []
		for a in children:
			eps = np.sqrt(np.log(N)/(a.N+1e-5))
			q_hat_ex = a.Q + c * eps
			r_hat_ex = a.R - lmda * eps
			val = q_hat_ex * float(r_hat_ex<=risk_bound)
			vals.append(val)
		return children[np.argmax(vals)]

	## Samples successor state from (s,a) based on visitation frequency
	def sample_successor_state(self,act_node):
		N = float(act_node.N-1)
		state_nodes = act_node.children
		probs = np.array([state_node.N/N for state_node in state_nodes])
		i = self.rng.choice(len(state_nodes),p=probs)
		return state_nodes[i]

	## Back up values up tree
	def backup(self,child_state_node,val,risk):
		action_node = child_state_node.parent
		q = val
		while action_node is not None:
			N = action_node.N
			q = child_state_node.r + self.discount * q
			action_node.Q += (q - action_node.Q) / N
			action_node.risk += (risk - action_node.risk) / N
			child_state_node = action_node.parent
			action_node = child_state_node.parent

	## Run rollout policy from leaf state until we reach the desired horizon
	def terminal_estimator(self,leaf,curr_depth):
		num_steps = self.horizon - curr_depth
		val,risk = self.rollout_policy(leaf.state,num_steps,self.discount,self.rng)
		if self.violates_constraint(leaf.state): risk = 1. ##CHECK: this should be done in rollout policy
		return val,risk

	## Prune all subtrees that violate chance constraints
	# def prune_tree(root,risk_bound,conservative=True):
	# 	states = [root]
	# 	while len(states>0):
	# 		curr_state = states.pop(0)
	# 		actions = curr_state.children
	# 		for i

	# 		for action in actions:
	# 			immediate_risk = action.immediate_risk



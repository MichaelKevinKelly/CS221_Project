from planner.nodes import DpwStateNode, DpwActionNode
from planner.bandits import bandit_rule_generator
import numpy as np

class RAMCTS(object):

	def __init__(self,
			env,
			exploration_type,
			niter = 5000,
			c = 3.0,
			lmbda = 3.0,
			rng = np.random.RandomState(1),
			k_s = 15.,
			k_a = 15.,
			alpha_s = 10.,
			alpha_a = 10.,
			p = 0.02,
			constant_ed = 1.,
			discount = 1.,
			return_most_simulated = False,
			risk_aware = True,
			single_step_risk_aware = True,
			conservative_risk_prop = True
		):

		## Problem-Specific Parameters
		self.env = env
		self.init_q = lambda state,act: 0
		
		## General MCTS Parameters
		self.niter = niter
		self.discount = discount
		self.return_most_simulated = return_most_simulated
		self.node_constants = {'k_s':k_s, 'k_a':k_a, 'alpha_s':alpha_s, 
			'alpha_a':alpha_a, 'exploration_type': exploration_type,
			'p': p, 'e_d': constant_ed}
		self.select_action = bandit_rule_generator(exploration_type, 
			{'c':c,'lmbda':lmbda, 'risk_aware':risk_aware})
		self.single_step_risk_aware = single_step_risk_aware
		self.conservative_risk_prop = conservative_risk_prop
	
		## Random Number Generator
		self.rng = rng
	
	## Plan from init_state to a horizon of self.horizon while respecting risk_bound
	def plan(self,init_state,risk_bound,horizon):
		assert(risk_bound>=0. and risk_bound<=1.)
		self.node_constants['dmax'] = horizon
		root = DpwStateNode(init_state,0.,None,0.,self.node_constants,self.env.violates_constraint(init_state),init_n=0)
		max_tree_depth = 0
		for it in range(self.niter):
			leaf, curr_depth = self.simulate(root,risk_bound,horizon)
			val, risk = self.env.rollout_policy(leaf.state,int(horizon-curr_depth),self.discount,self.rng)
			self.backup(leaf,val,risk)
			max_tree_depth = max(max_tree_depth,curr_depth)
		act = self.select_single_action(root,risk_bound)
		print('Max tree depth: {}'.format(max_tree_depth))
		return (None,None) if act is None else (act.act,act.immediate_risk)

	## Choose single action to execute from root state
	def select_single_action(self,root,risk_bound):
		actions = root.children
		if self.single_step_risk_aware:
			actions = [a for a in actions if a.risk<=risk_bound]
		if self.return_most_simulated:
			action_vals = [a.N for a in actions]
		else:
			action_vals = [a.Q for a in actions]
		return None if len(actions)==0 else actions[np.argmax(action_vals)]

	## Performs rollout of tree policy
	def simulate(self,root,risk_bound,horizon):
		root.N += 1
		curr_node, curr_risk_bound, d = root, risk_bound, 0
		_break = False ##TODO: should we recurse on states that violate the constraint? No...--> we shouldn't!! see RAO* paper, limitation of assuming violating state is terminal
		while d<horizon and not _break: 
			# print(d)
			act_node, curr_risk_bound = self.action_pw(curr_node,curr_risk_bound,d+0.5)
			curr_node, _break = self.state_pw(act_node,d+1.)
			d += 1
		return curr_node, d

	## Implements action progressive widening (see https://hal.archives-ouvertes.fr/hal-00542673v1/document)
	def action_pw(self,state_node,curr_risk_bound,depth):
		## Add a new action if allowed
		created_node = False
		if state_node.allow_new_node:
			act = self.env.random_act_generator(state_node.state,self.rng)
			new_act_node = DpwActionNode(act,state_node,depth,self.node_constants,
				init_n=0,init_q=self.init_q(state_node.state,act),risk=0.)
			state_node.children.append(new_act_node)
			created_node = True
		else:
			# print('Maxed out actions')
			pass

		## Choose from among existing actions acording to risk-ucb
		act_node = self.select_action(state_node,curr_risk_bound)
		# if created_node:
		# 	assert(len(act_node.children)==0)
		# else:
		# 	assert(len(act_node.children)>0)
		updated_risk_bound = self.propagate_risk_bound(act_node, curr_risk_bound) if len(act_node.children)>0 else None
		act_node.N += 1
		return act_node, updated_risk_bound

	## Implements state progressive widening (see https://hal.archives-ouvertes.fr/hal-00542673v1/document)
	def state_pw(self,act_node,depth):
		## Sample a new successor state if allowed
		init_state_node = act_node.parent
		_break = False
		if act_node.allow_new_node:
			state, r = self.env.dynamics(init_state_node.state,act_node.act,self.rng)
			state_node = DpwStateNode(state,r,act_node,depth,self.node_constants,
				self.env.violates_constraint(state),init_n=1)
			act_node.children.append(state_node)
			_break = True
		else:
			# print('Maxed out states')
			# print(act_node.N)
			# print(act_node.k*act_node.N**act_node.alpha)
			# print(len(act_node.children))
			# print()
			# print(act_node.k)
			# print(act_node.alpha)
			# print()
			state_node = self.sample_successor_state(act_node)
			state_node.N += 1
		# self.propagate_risk_bound(act_node,state_node)
		if state_node.violates_constraint:
			# _break = True ##TODO: justify this (at this point we've violated constraints with p=1> any reasonable threshold) -> don't stop execution though!
			act_node.N_immediate_violations += 1
		return state_node, _break

	## Propagates risk bounds from an action node to one of its children
	def propagate_risk_bound(self, act_node, curr_risk_bound):
		
		# print(act_node.parent.parent)
		# print(act_node.children)
		# print(act_node.N)
		
		if self.conservative_risk_prop:
			####### Check ###### #TODO: delete check
			N = float(act_node.N)
			total_one_step_failures = sum([s.N for s in act_node.children if s.violates_constraint])
			assert(abs(act_node.immediate_risk-total_one_step_failures/N)<1e-5)
			####################
			updated_risk_bound = curr_risk_bound - act_node.immediate_risk
		else:
			raise NotImplementedError
		return updated_risk_bound
		
	## Samples successor state from (s,a) based on visitation frequency
	def sample_successor_state(self,act_node):
		N = float(act_node.N-1) ##TODO: why -1? -> because we've already incremented N for act_node but havent reached successor state yet
		state_nodes = act_node.children
		probs = np.array([state_node.N/N for state_node in state_nodes])
		i = self.rng.choice(len(state_nodes),p=probs) ##TODO: alternative ways of doing this? Also isn't this just uniform sampling?
		return state_nodes[i]

	## Back up values up tree ##TODO: Is r just binary? Why not backup q*r?-> doesn't reason about threshold -> hard constraint, don't care about risk value provided it satisfies threshold
	def backup(self,child_state_node,q,risk):
		action_node = child_state_node.parent
		while action_node is not None:
			N = action_node.N
			q = child_state_node.reward + self.discount * q
			action_node.Q += (q - action_node.Q) / N
			action_node.risk += (risk - action_node.risk) / N ##TODO: is this the right way to do this?? (maybe, risk isn't discounted....)
			assert(action_node.risk>=0. and action_node.risk<=1.)
			child_state_node = action_node.parent
			action_node = child_state_node.parent


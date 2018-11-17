import numpy as np
from nodes import DpwStateNode, DpwActionNode

class MCTS(object):

	def __init__(self,
			dynamics,
			random_act_generator,
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
		self.terminal_estimator = terminal_estimator
		self.random_act_generator = random_act_generator
		
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
			d = max(d,_d)
			val = self.terminal_estimator(leaf,self.discount,self.rng)
			self.backup(leaf,val)
		act = self.select_action(root,0).act
		print('Max depth: {}'.format(d))
		return act

	## Determine in-tree simulation
	def simulate(self,root):
		root.N += 1
		curr_node, d = root, 0
		new_state = False
		while d<self.max_depth and not new_state:
			act_node = self.action_pw(curr_node)
			curr_node, new_state = self.state_pw(act_node)
			d += 1
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
	def state_pw(self,act_node):
		## Sample a new successor state if allowed
		init_state_node = act_node.parent
		new = False
		if len(act_node.children)<act_node.max_children:
			state, r = self.dynamics(init_state_node.state,act_node.act,self.rng)
			state_node = DpwStateNode(state,act_node,r,self.k_s,self.alpha_s,init_n=self.init_n)
			act_node.children.append(state_node)
			new = True
		else:
			# print('Maxed out states')
			state_node = self.sample_successor_state(act_node)
		state_node.N += 1
		return state_node, new

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
			action_node.Q += (q - action_node.Q) / action_node.N
			i+=1
			child_state_node = action_node.parent
			action_node = child_state_node.parent

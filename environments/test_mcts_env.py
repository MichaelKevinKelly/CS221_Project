from environments.abstract_env import AbstractEnv
import numpy as np

class BasicMCTSTestEnv(AbstractEnv):

	'''Generate next state after taking action 'act' from state 'state', using provided rng'''
	def dynamics(self,state,act,rng):
		nxt_state = state + act
		
		if nxt_state<1.:
			r = 10
		elif nxt_state<2.:
			r = 0
		else:
			# print(state)
			# print(act)
			# print(nxt_state)
			# assert(False)
			r = 100
		return nxt_state, r
		# return state + np.random.rand(), 0.
		
	'''Perform rollout for num_step'''
	'''Returns the return ('val') and a bit ('risk') indicating if constraints were violated'''
	def rollout_policy(self,state,num_steps,discount,rng):
		return 0,0
		# violation, q = self.violates_constraint(state), 0.
		# for i in range(num_steps):
		# 	act = rng.rand()
		# 	state, r = self.dynamics(state,act,rng)
		# 	q += discount**i * r
		# 	violation |= self.violates_constraint(state)
		# return q, violation
		
	'''Generates a random action from state 'state' using the rng'''
	def random_act_generator(self,state,rng):
		return rng.rand()-0.1
	
	'''Checks whether a given state violates a (deterministic) constraint'''
	def violates_constraint(self,state):
		return state<-10000
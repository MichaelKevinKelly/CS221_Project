import abc

class AbstractEnv(metaclass=abc.ABCMeta):

	@staticmethod
	@abc.abstractmethod
	def dynamics(self,state,act,rng):
		'''Generate next state & reward after taking action 'act' from state 'state', using provided rng'''
		'''All rewards should be in the range (0,1)'''
	
	@staticmethod
	@abc.abstractmethod
	def rollout_policy(self,state,num_steps,discount,rng):
		'''Perform rollout for num_step'''
		'''Returns the return ('val') and a bit ('risk') indicating if constraints were violated'''

	@staticmethod
	@abc.abstractmethod
	def random_act_generator(self,state,rng):
		'''Generates a random action from state 'state' using the rng'''

	@staticmethod
	@abc.abstractmethod
	def violates_constraint(self,state):
		'''Checks whether a given state violates a (deterministic) constraint'''
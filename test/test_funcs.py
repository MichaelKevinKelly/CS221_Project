from planner.ar_mcts_v1 import RAMCTS
from planner.nodes import DpwStateNode, DpwActionNode
from environments.test_mcts_env import BasicMCTSTestEnv
import numpy as np
import yaml

params_path = 'mcts_test_parameters.yaml'
params = yaml.load(open(params_path))['parameters']
test_env = BasicMCTSTestEnv()

mcts = RAMCTS(test_env,
		params['exploration_type'],
		niter = params['niter'],
		c = params['c'],
		lmbda = params['lmbda'],
		rng = np.random.RandomState(params['random_seed']),
		k_s = params['k_s'],
		k_a = params['k_a'],
		alpha_s = params['alpha_s'],
		alpha_a = params['alpha_a'],
		p = params['p'],
		constant_ed = params['constant_ed'],
		discount = params['discount'],
		return_most_simulated = params['return_most_simulated'],
		risk_aware = params['risk_aware'],
		single_step_risk_aware = params['single_step_risk_aware'],
		conservative_risk_prop = params['conservative_risk_prop'])

## Create node constants
node_constants = {'k_s':params['k_s'], 'k_a':params['k_a'], 'alpha_s':params['alpha_s'], 'alpha_a':params['alpha_a'],
			'dmax': params['mcts_horizon'], 'exploration_type': params['exploration_type'], 'p': params['p'], 
			'e_d': params['constant_ed']}

## State node data
state = 0.
reward = 0.
parent = None
depth = 0.
init_n = 1
init_q = 0
risk = 0

## Generate state node
state_node = DpwStateNode(state,reward,parent,depth,node_constants,test_env.violates_constraint(state),init_n=init_n)

print('Running tests....')

## Checks for select single action

## No children
risk_bound=1.
assert(mcts.select_single_action(state_node,risk_bound) is None)

## One child (satisfies risk constraint)
act = 0.
parent = state_node
depth = 0.5
child_node = DpwActionNode(act,parent,depth,node_constants,init_n=0,init_q=0,risk=0.)
state_node.children.append(child_node)
assert(mcts.select_single_action(state_node,risk_bound) is not None)
assert(mcts.select_single_action(state_node,risk_bound).act == act)

## One child (violates risk constraint)
risk_bound=0.01
child_node.risk=0.1
assert(mcts.select_single_action(state_node,risk_bound) is None)

## Multiple children (satisfy risk constraint)
child_node.risk=0.01
risk_bound=0.1
for act in range(1,5):
	curr_child = DpwActionNode(float(act),parent,depth,node_constants,init_n=0,init_q=0,risk=0.01)
	state_node.children.append(curr_child)
assert(mcts.select_single_action(state_node,risk_bound) is not None)
assert(mcts.select_single_action(state_node,risk_bound).act == 0)

## Multiple children (violate risk constraint)
for child in state_node.children:
	child.risk = 1.
assert(mcts.select_single_action(state_node,risk_bound) is None)

## Multiple children (only act==3) satisfies risk constraint
state_node.children[3].risk=0.01
assert(mcts.select_single_action(state_node,risk_bound) is not None)
assert(mcts.select_single_action(state_node,risk_bound).act==3.)

## Clear children
state_node.children = []
assert(mcts.select_single_action(state_node,risk_bound) is None)

## Test single action pw
risk_bound = 0.1
mcts.action_pw(state_node,risk_bound,0+0.5)
assert(mcts.select_single_action(state_node,risk_bound) is not None)
init_act = mcts.select_single_action(state_node,risk_bound).act

## Add a number of new children
for i in range(10):
	state_node.N += 1
	print('-------')
	print(state_node.N)
	print(state_node.alpha)
	print(np.floor(state_node.N**state_node.alpha))
	print(np.floor((state_node.N-1)**state_node.alpha))
	print('State_node.N: {}'.format(state_node.N))
	# print('Max children: {}'.format(state_node.get_max_children_ka))
	act_node, rb = mcts.action_pw(state_node,risk_bound,0+0.5)
	print('Num children state_node: {}'.format(len(state_node.children)))
	assert(state_node.N >= len(state_node.children))
	assert(state_node.N==sum(a.N for a in state_node.children))
	print()
for act in state_node.children:
	act.risk = 0.3
assert(mcts.select_single_action(state_node,risk_bound) is None)

## Choose single valid child
state_node.children[2].risk=0.
act_node = mcts.select_single_action(state_node,risk_bound)
assert(act_node is not None)

state, _break = mcts.state_pw(act_node,0.+1.)
print(state.state)
print(state.reward)
print(_break)

## Finish checking state_pw


## Test propagate risk bound
state_node.children = []
state_node.N = 1
risk_bound = 1.

print('\nIteration 1')
act_node, rb = mcts.action_pw(state_node,risk_bound,0.+0.5)
print(rb)
child_state_node, _break = mcts.state_pw(act_node,0.+1.)
assert(_break)

print('\nIteration 2')
state_node.N += 1
act_node.N += 1
child_state_node, _break = mcts.state_pw(act_node,0.+1.)
assert(_break)

print('\nIteration 2')
state_node.N += 1
act_node.N += 1
child_state_node, _break = mcts.state_pw(act_node,0.+1.)
assert(_break)

print('\nAction\'s children')
for s in act_node.children:
	print(s.state)
	print(s.violates_constraint)
print(act_node.immediate_risk)

# act_node, risk_bound = mcts.action_pw(state_node,risk_bound,0.+0.5)
print('New risk bound: {}'.format(mcts.propagate_risk_bound(act_node,risk_bound)))

# ## Test backup 
print('\nTesting backup (single iteration)')
assert(state_node.N==act_node.N==sum([s.N for s in act_node.children]))
assert(act_node.Q==0 and act_node.risk==0)
s = act_node.children[0]
mcts.discount=0.5
mcts.backup(s,1.,1/3.)
assert(abs(act_node.Q-1./6.)<1e-5 and abs(act_node.risk-1./9.)<1e-5)


# ## Test rollout policy
print('\nTesting rollout policy')
# q,r = mcts.env.rollout_policy(0.,10,discount=1.,rng=mcts.rng)
# assert(r==0)

# q,r = mcts.env.rollout_policy(0.,5,discount=0.9,rng=mcts.rng)
# q_pred = 0
# for i in range(1,6):
# 	q_pred += 0.9**(i-1) * i
# assert(abs(q-q_pred)<1e-5)
# assert(r==0)

# q,r = mcts.env.rollout_policy(0.,5,discount=0.9,rng=mcts.rng)
# assert(q==0.)
# assert(r==1.)

# ## Test simulate (short horizon)

# ## Test plan (limited niter and horizon)




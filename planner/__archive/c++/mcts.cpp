#include "mcts.hpp"

MCTS::MCTS(*MCTSParameters p){
	dynamics = p.env.dynamics;
	terminal_estimator = p.env.terminal_estimator;
	rand_act_generator = p.env.rand_act_generator;
	rng = p.env.rng; //
	niter = p.niter;
	max_depth = p.max_depth;
	c = p.c;
	discount = p.discount;
	init_q = p.init_q;
	init_n = p.init_n;
	k_s = p.k_s;
	k_a = p.k_a;
	alpha_s = p.alpha_s;
	alpha_a = p.alpha_a;	
}

Act* MCTS::plan(const State init_state){
	StateNode& root = StateNode(init_state,NULL,0,k_s,alpha_s,init_n)
	for (int i = 0; i < niter; i++) {
		StateNode& leaf = simulate(root)
		double val = terminal_estimator(leaf,discount,rng)
		backup(leaf,val)
	}
	//TODO: Clean up tree
	return select_action(root,0).act
}

StateNode& MCTS::simulate(StateNode& root){
	root.N = root.N + 1;
	StateNode& curr_node = root;
	int d = 0;
	while root.
}


*ActNode action_pw(*StateNode state_node, double c);
*StateNode state_pw(*ActNode act_node);
*ActNode select_action(*StateNode state_node, double c);
*StateNode sample_successor_state(*ActNode act_node);
void backup(*StateNode leaf_node, double val);
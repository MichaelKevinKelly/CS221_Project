#pragma once

#include <vector>
#include "state.hpp"
#include "mcts_parameters.hpp"
#include "environment.hpp"
#include "nodes.hpp"

namespace mcts{
	
	class MCTS{
		public:
			MCTS(const MCTSParameters& p);
			Act* plan(State init_state);
			~MCTS();
		private:
			StateNode& simulate(StateNode& root);
			ActNode& action_pw(StateNode& state_node, double c);
			StateNode& state_pw(ActNode& act_node);
			ActNode& select_action(StateNode& state_node, double c);
			StateNode& sample_successor_state(ActNode& act_node);
			void backup(StateNode& leaf_node, double val);
			int niter;
			int max_depth;
			double c;
			double discount;
			double init_q;
			int init_n;
			double k_s;
			double k_a;
			double alpha_s;
			double alpha_a;
			Dynamics dynamics;
			TerminalEstimator terminal_estimator;
			RandomActGenerator rand_act_generator;
			RandomNumberGenerator rng;
	}
}

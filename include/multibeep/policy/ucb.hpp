#ifndef MULTIBEEP_POLICY_UCB
#define MULTIBEEP_POLICY_UCB

#include <iostream>

#include "multibeep/policy/policy.hpp"
#include "multibeep/bandit/bandit.hpp"

namespace multibeep{ namespace policies{
	template<typename num_t=double, typename rng_t = std::default_random_engine>
	class UCB_base: public multibeep::policies::base<num_t, rng_t>{
		protected:
			typedef multibeep::policies::base<num_t, rng_t> base_t;
			
			/* \brief allows for various UCB flavours that only differ in the definition of the confidence gap
			 * 
			 * UCB algorithms usually pick arm i^* according to
			 * 
			 * i^* = argmax_{i=0...k} (mu_i + g_i)
			 * 
			 * where g_i is the confidence gap (usually the confidence intervall for the mean estimate).
			 * 
			 * The function must return NAN if computing the gap failed because the arm was not pulled often
			 * enough. This return value trigger this arm to be pulled next.
			 * */
			virtual num_t calculate_confidence_gap(const multibeep::bandits::arm_info<num_t, rng_t> &ai) =0;
			
		std::shared_ptr<rng_t> rng_ptr;
			
		public:
			UCB_base(std::shared_ptr<multibeep::bandits::base<num_t, rng_t> > b_ptr, std::shared_ptr<rng_t> r_ptr):
				base_t(b_ptr), rng_ptr(r_ptr){}
			
			virtual unsigned int select_next_arm(){
				
					unsigned int best_index;
					num_t rnd = std::numeric_limits<num_t>::lowest();
					num_t max_ucb = std::numeric_limits<num_t>::lowest();
				
					for (auto i=0u; i < base_t::bandit_ptr->number_of_active_arms(); i++){
						
						auto & ai = (*base_t::bandit_ptr)[i];
						
						num_t gap =  calculate_confidence_gap(ai);
						// if there was not enough information to compute the gap yet, pull this one
						if (std::isnan(gap))
							return(i);
						
						num_t ucb = ai.estimated_mean + gap;
						// pick a random index for equivalent values
						if (ucb == max_ucb){
							
							// draw random number
							num_t tmp = (*rng_ptr)();
							
							// keep index with the largest random number
							// should be equivalent to picking one at random
							if (tmp > rnd){
								rnd = tmp;
								best_index = i;
							}
						}
						
						if (ucb > max_ucb){
							max_ucb = ucb;
							rnd = (*rng_ptr)();
							best_index = i;
						}
					}
				
				return(best_index);
			}
	};
	
	
	
	
/*	
	template<class BanditClass, class Engine>
	class PolicyUCB: public Policy<BanditClass>{
	protected:
		virtual double calculate_confidence_gap(const ArmInfo& arm_info) = 0;
		Engine& rng;
	public:
		PolicyUCB(BanditClass& bandit, Engine& rng): Policy<BanditClass>(bandit), rng(rng){};
		virtual std::string  get_ident() = 0;
		
		const ArmInfo& select_next_arm(){
			//pull all arms at least once
			std::pair<bool, unsigned int> hast_to_pull = this->pull_at_least(this->num_pull_at_least, rng);
			if (hast_to_pull.first){
				return this->bandit.get_arm_infos()[hast_to_pull.second];
			}
			std::vector<double> ucb_values;
			std::vector<unsigned int> idx;
			for(auto it = this->bandit.begin(); it !=this->bandit.end(); ++it){
				double mean = (*it).sample_mean;
				if (this->bandit.is_model_used()){
					mean = (*it).estimated_mean;
				}
				double ucb = mean +  calculate_confidence_gap(*it);
				ucb_values.push_back(ucb);
				idx.push_back((*it).ident);
			}
			unsigned int selected = this->select_max_idx(ucb_values, idx, rng);
			return this->bandit.get_arm_infos()[selected];
			
		};
	};
*/
}}
#endif

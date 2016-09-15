#ifndef MULTIBEEP_POLICY_SUCCESSIVE_HALVING
#define MULTIBEEP_POLICY_SUCCESSIVE_HALVING

#include <stdexcept>

#include "multibeep/policy/policy.hpp"
#include "multibeep/bandit/bandit.hpp"

namespace multibeep{ namespace policies{

	template<typename num_t=double, typename rng_t = std::default_random_engine>
	class successive_halving: public multibeep::policies::base<num_t, rng_t>{
		protected:
			typedef multibeep::policies::base<num_t,rng_t> base_t;
			
			
			unsigned int min_pulls_per_round;
			
			num_t eta_pulls;
			num_t eta_arms;
			
		public:
		
			successive_halving(std::shared_ptr<multibeep::bandits::base<num_t, rng_t> > b_ptr,
								unsigned int min_pulls, num_t eta_arms, num_t eta_pulls):
				base_t(b_ptr), min_pulls_per_round(min_pulls), eta_pulls(eta_pulls), eta_arms(eta_arms) {}
		
		
			successive_halving(std::shared_ptr<multibeep::bandits::base<num_t,rng_t> > b_ptr,
								unsigned int min_pulls, num_t frac):
				successive_halving(b_ptr, min_pulls, frac,frac) {}
			
			std::string get_ident() {return(std::string("successive halfing"));}
			
			virtual unsigned int select_next_arm(){
				throw std::runtime_error("Successive Halving cannot select a next arm. Use the play_n_rounds method to run it for a fixed budget");
			}
			
			
			virtual void play_n_rounds (unsigned int num_rounds){
				
				// convenience alias for the bandit
				auto &b (*(base_t::bandit_ptr));
				
				unsigned int r_k = min_pulls_per_round;
				
				for (int round=0; round< (int) num_rounds; round++){
					
					for (auto mew=0u; mew < b.number_of_active_arms(); mew++){
						for (auto mew2=0u; mew2 < r_k; mew2++)
							b.pull_by_index(mew);
					}
					// deactivate all obsolete arms
					
					int n_kp1 = std::max(1., std::round( b.number_of_active_arms()/eta_arms));
					r_k *= eta_pulls;
					b.deactivate_n_worst( b.number_of_active_arms() - n_kp1);
				}
				
			}
			
	};

}}
#endif

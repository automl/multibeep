#ifndef MULTIBEEP_POLICY_RANDOM
#define MULTIBEEP_POLICY_RANDOM

#include <random>

#include "multibeep/policy/policy.hpp"
#include "multibeep/bandit/bandit.hpp"

namespace multibeep{ namespace policies{
	
	template<typename num_t=double, typename rng_t = std::default_random_engine>
	class random: public multibeep::policies::base<num_t, rng_t>{
		protected:
			typedef multibeep::policies::base<num_t, rng_t> base_t;
			std::shared_ptr<rng_t> rng_ptr;
		public:
			random(std::shared_ptr<multibeep::bandits::base<num_t, rng_t> > b_ptr, std::shared_ptr<rng_t> r_ptr):
				base_t(b_ptr), rng_ptr(r_ptr){}
			
			std::string  get_ident(){ return std::string("Random Policy");}
		
		/* \brief  random search just returns a random index in the range 0 to the number of active arms - 1*/
		virtual unsigned int select_next_arm(){
			std::uniform_int_distribution<unsigned int> u (0,base_t::bandit_ptr->number_of_active_arms()-1);
			return(u(*(rng_ptr)));
			
		}
	};
}}
#endif

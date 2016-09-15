#ifndef MULTIBEEP_POLICY_UCBP
#define MULTIBEEP_POLICY_UCBP

#include "multibeep/policy/ucb.hpp"
#include "multibeep/bandit/bandit.hpp"


namespace multibeep{ namespace policies{
	template<typename num_t=double, typename rng_t = std::default_random_engine>
	class UCB_p: public UCB_base<num_t, rng_t>{
	protected:
		typedef multibeep::policies::base<num_t, rng_t> base_t;
		
		num_t p;
		num_t calculate_confidence_gap(const multibeep::bandits::arm_info<num_t, rng_t> &ai){
			
			if (std::isnan(ai.estimated_variance))	return(NAN);
			
			auto mean_variance = ai.estimated_variance;
			auto N = base_t::bandit_ptr->number_of_pulls();
			/* the original algorithm calls for 
			 * 
			 *			std::sqrt((variance/n*p*log(N)));
			 * 
			 * where variance is the variance of the reward distribution and n is the number of pull for that arm.
			 * In multibeep "estimated_variance" is the variance for the estimate of the mean, corresponding to variance/n.
			 */
			return std::sqrt((mean_variance*p*log(N)));
		};
	public:
		UCB_p(std::shared_ptr<multibeep::bandits::base<num_t, rng_t> > b_ptr, std::shared_ptr<rng_t> r_ptr, num_t p):
			multibeep::policies::UCB_base<num_t, rng_t>(b_ptr, r_ptr),p(p){}
		
		std::string  get_ident(){ return std::string("UCBp");}
	};
}}
#endif

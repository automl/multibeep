#ifndef MULTIBEEP_POLICY_UCBP
#define MULTIBEEP_POLICY_UCBP

#include "multibeep/policy/ucb.hpp"
#include "multibeep/bandit/bandit.hpp"


namespace multibeep{ namespace policies{

    /* Another UCB variant with logarithmic regret that explicitly uses variance estimates
     *
     * Reference:
     * Audibert, Munos, Szepesvári: Exploration-exploitation trade-off using variance estimates in multi-armed bandits. Theoretical Computer Science Volume 410 Issue 19 Apr. 2009 pp. 1876-1902
     */
	template<typename num_t=double, typename rng_t = std::default_random_engine>
	class UCB_V: public UCB_base<num_t, rng_t>{
	protected:
		typedef multibeep::policies::base<num_t, rng_t> base_t;
		
		num_t b;
        num_t zeta;

        
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
			return std::sqrt(2*mean_variance*std::log(N)) + b*std::log(N)/ai.num_pulls;
		};
	public:
		UCB_p(std::shared_ptr<multibeep::bandits::base<num_t, rng_t> > b_ptr, std::shared_ptr<rng_t> r_ptr, num_t zeta, num_t b):
			multibeep::policies::UCB_base<num_t, rng_t>(b_ptr, r_ptr),b(b), zeta(zeta){}
		
		std::string  get_ident(){ return std::string("UCBV");}
	};
}}
#endif

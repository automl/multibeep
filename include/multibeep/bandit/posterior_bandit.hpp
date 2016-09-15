#ifndef MULTIBEEP_POSTERIOR_BANDIT
#define MULTIBEEP_POSTERIOR_BANDIT

#include "multibeep/bandit/bandit.hpp"


namespace multibeep{ namespace bandits{

template <typename num_t = double, typename rng_t = std::default_random_engine>
class posterior: public base<num_t, rng_t>{
	
	typedef base<num_t, rng_t> base_t;
	
	public:
		virtual void update_arm_info(unsigned int index){
			auto &ai = base_t::arm_infos.at(index);

			if (ai.dirty){
				ai.posterior = ai.get_arm_ptr()->posterior();
				if (ai.posterior){ // only provide a mean and a variance if the posterior is valid
					ai.estimated_mean = ai.posterior->mean();
					ai.estimated_variance = ai.posterior->variance();
				}
				ai.dirty=false;
			}
		}
};


}} // namespaces
#endif

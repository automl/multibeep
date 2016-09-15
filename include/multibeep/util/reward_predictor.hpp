#ifndef MULTIBEEP_UTIL_REWARD_PREDICTOR
#define MULTIBEEP_UTIL_REWARD_PREDICTOR

#include <random>

namespace multibeep{ namespace util{ namespace reward_predictor{


/* \brief general interface for reward fortune tellers*/
template <typename num_t = double>
class base{
	/* \brief foresee a random reward */
	virtual num_t operator() () = 0;
};





/* \brief template class to leverage all sort of random distributions from std and boost*/
template <typename rnd_dist_t,typename rng_t, typename num_t = double>
class random_distribution: public base<num_t>{
	
	protected:
		rnd_dist_t rnd_dist;
	
	public:
		random_distribution( rnd_dist_t rd): rnd_dist(rd) {}
		
		virtual num_t operator() (rng_t & rng){
			return(rnd_dist(rng));
		}
};


/* \brief prediction is a previously obtained reward */
template <typename arm_info_t, typename rng_t, typename num_t = double>
class history: public base<num_t>{
	
	protected:
		const arm_info_t & arm;
	
	public:
		history (arm_info_t & ai): arm(ai) {}
		
		virtual num_t operator() (rng_t& rng){
			std::uniform_int_distribution<unsigned int> dist(0, arm.rewards.size());
			return(arm.rewards[dist(rng)]);
		}
};



/* \brief prediction is a previously obtained reward */
template <typename bandit_t, typename arm_info_t, typename rng_t, typename num_t = double>
class history_mix_arms: public base<num_t>{
	
	protected:
		const bandit_t & bandit;
		const arm_info_t & arm;
		num_t alpha;
	
	public:
		history_mix_arms(bandit_t & b, arm_info_t & ai, num_t a):
			bandit(b), arm(ai), alpha(a) {}
		
		virtual num_t operator() (rng_t& rng){
			std::uniform_real_distribution<num_t> u_dist(0,1);
			
			num_t u = udist(rng);
			
			const arm_info_t & tmp_arm = arm;
			
			
			// check wether we should draw a sample from any other arm
			if (u < 2* alpha/ (arm.num_pulls + 2* alpha)){
				
				// choose any of the already pulled arms
				std::uniform_int_distribution<unsigned int> i_dist(1,bandit.number_of_pulled_arms());
				
				unsigned int i = i_dist(rng);
				unsigned int j = 0;
				
				// find the i^th pulled arm
				do{
					while (bandit[j].rewards.size()==0){
						j++;
					}
					i--;
				}while(i > 0);
				tmp_arm = bandit[j];
			}
			std::uniform_int_distribution<unsigned int> i_dist(0,tmp_arm.rewards.size()-1);
			return(tmp_arm.rewards[i_dist(rng)]);			
		}
};



}}}
#endif

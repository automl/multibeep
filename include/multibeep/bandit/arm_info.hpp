#ifndef MULTIBEEP_BANDIT_ARM_INFO
#define MULTIBEEP_BANDIT_ARM_INFO

#include <memory>
#include <vector>
#include <iostream>

#include <boost/math/distributions/normal.hpp>

#include <multibeep/util/statistics.hpp>
#include <multibeep/util/posteriors.hpp>
#include <multibeep/util/reward_predictor.hpp>

#include <multibeep/arm/arm.hpp>

namespace multibeep{ namespace bandits {



template <typename num_t = double, typename rng_t = std::default_random_engine>
class arm_info{
	protected:
		std::shared_ptr<multibeep::arms::base<num_t, rng_t> > arm_ptr;
	public:
	
		/* \brief a unique identifier for each arm*/
		unsigned int identifier;
		/* \brief indicator whether the arm is considered active*/
		bool is_active;
		/* \brief indicator whether the arm info needs to be updated*/
		bool dirty;
		
		/*\brief number of pulls for this arm. This could be an estimated number if a model-based bandit is used*/
		long double num_pulls;
		/* \brief keeps track of the reward mean and variance */
		multibeep::util::statistics::running_statistics<num_t> reward_stats;
		/* \brief all previously received rewards, if the appropriate bandit is used*/
		std::vector<num_t> rewards;
		
		/* \brief probability that this arm has the highest mean reward*/
		num_t p_max;
		
		/* \brief access to the arms posterior via a pointer*/
		std::shared_ptr<multibeep::util::posteriors::base<num_t, rng_t> > posterior;
		
		/*\brief mean estimate to be used by the policies
		 * 
		 * Different bandits will assign different mean estimates here.
		 * Examples are the empirical mean, the mean of the arm's posterior
		 * or a robust estimate.
		 */
		num_t estimated_mean;

		/* \brief variance of the estimated mean, i.e. a confidence interval for the mean estimation.
		 *
		 * Different bandits will assign different estimates here.
		 * Examples are the standard error of the mean, the variance of
		 * the arm's posterior or a robust estimate.
		 */
		num_t estimated_variance;


		arm_info(std::shared_ptr<multibeep::arms::base<num_t, rng_t> >ptr, unsigned int ident): 
			arm_ptr(ptr),
			identifier(ident),
			is_active(true),
			dirty(true),
			num_pulls(0),
			reward_stats(),
			rewards(),
			p_max(NAN),
			estimated_mean(NAN),
			estimated_variance(NAN)
			{};

		/* \brief pulls the underlying arm and updates the reward statistics
		 * 
		 * Pulls and updates the rewards history and statistics*/
		virtual double pull(){
			double r = arm_ptr->pull();
			reward_stats(r);
			rewards.push_back(r);
			num_pulls++;
			dirty = true;
			return(r);
		}

		/*\brief access to the arm pointer for arm specific information
		 * 
		 * Note that the returned pointer is const, meaning the arm cannot
		 * be pulled. This ensures that the arm can only be pulled via the
		 * bandit's pull method.
		 */
		std::shared_ptr<const multibeep::arms::base<num_t, rng_t> > get_arm_ptr() const {return(arm_ptr);}
	
};


/* I don't think I need a copy constructor at this point
 * 		ArmInfo(const ArmInfo& amr_info):sample_mean(amr_info.sample_mean), 
			sample_variance(amr_info.sample_variance), 
			robust_variance(amr_info.robust_variance),
			estimated_variance(amr_info.estimated_variance), 
			estimated_mean(amr_info.estimated_mean), 
			estimated_mean_variance(amr_info.estimated_mean_variance),
			number_pulled(amr_info.number_pulled), 
			sum_reward(amr_info.sum_reward), 
			sum_squared_reward(amr_info.sum_squared_reward), 
			is_active(amr_info.is_active), 
			ident(amr_info.ident), 
			arm(amr_info.arm),
			trajectory(amr_info.trajectory)
			{};
*/

/*		static bool is_sample_mean_bigger(const ArmInfo& ai1, const ArmInfo& ai2){
			if (ai1.number_pulled > 0 && ai2.number_pulled > 0)
				return  ai1.sample_mean > ai2.sample_mean;
			if (ai1.number_pulled > 0 && ai2.number_pulled == 0)
				return true;
			return false;
		}
		static bool is_estimated_mean_bigger(const ArmInfo& ai1, const ArmInfo& ai2){
			if (ai1.number_pulled > 0 && ai2.number_pulled > 0)
				return  ai1.estimated_mean > ai2.estimated_mean;
			if (ai1.number_pulled > 0 && ai2.number_pulled == 0)
				return true;
			return false;
		}
*/
/*		ArmInfo(): sample_mean(0), sample_variance(0), robust_variance(0), estimated_variance(0), estimated_mean(0), estimated_mean_variance(0),
			number_pulled(0), sum_reward(0), sum_squared_reward(0), is_active(true), 
			ident(-1){};
*/




}}

#endif

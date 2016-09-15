#ifndef MULTIBEEP_RECOMMENDER
#define MULTIBEEP_RECOMMENDER


#include <vector>
#include <random>


namespace multibeep{ namespace recommender{


	template<typename num_t = double, typename rng_t = std::default_random_engine>
	class base{

		/* \brie computes the instantanious regret based on the recommendation distribution*/
		num_t instantanious_regret (multibeep::bandits::base * b_ptr) = 0;

		/* \brief suggests a distribution over the arms by index*/
		std::vector<num_t> recommender_distribution_index (multibeep::bandits::base * b_ptr, bool consider_inactive = false) = 0;

		/* \brief suggests a distribution over all arms by identifier*/
		std::vector<num_t> recommender_distribution_identifier (multibeep::bandits::base * b_ptr) = 0;

	};

}}
#endif

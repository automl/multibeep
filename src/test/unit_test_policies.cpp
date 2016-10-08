#include <random>

#include <boost/test/unit_test.hpp>

#include "multibeep/policy/random.hpp"
#include "multibeep/policy/prob_match.hpp"
#include "multibeep/policy/ucbp.hpp"
#include "multibeep/policy/successive_halving.hpp"


#include "multibeep/bandit/empirical_bandits.hpp"
#include "multibeep/bandit/posterior_bandit.hpp"
#include "multibeep/arm/normal.hpp"



typedef boost::random::mt19937 rng_t;
typedef double num_t;


template <typename policy_t, typename bandit_t, typename ... T>
void test (unsigned int num_arms, unsigned int num_rounds, unsigned int expected_num_pulls, T...t){

	std::shared_ptr<bandit_t> bandit_ptr = std::make_shared<bandit_t> (bandit_t());
	policy_t policy(bandit_ptr,  t...);


	std::shared_ptr<rng_t> rng_ptr = std::make_shared<rng_t> (rng_t () );
		
	std::uniform_real_distribution<num_t> u (-1,1);
		
	for (auto mew =0u; mew < num_arms; mew++)
		bandit_ptr->add_arm(std::shared_ptr<multibeep::arms::base<num_t, rng_t> > (new multibeep::arms::normal_arm<num_t,rng_t> (u(*rng_ptr),u(*rng_ptr)+1, rng_ptr)));


	policy.play_n_rounds(num_rounds);

	BOOST_REQUIRE_EQUAL(bandit_ptr->number_of_pulls(), expected_num_pulls);

}




BOOST_AUTO_TEST_CASE(test_random_policy){

	std::shared_ptr<rng_t> rng_ptr = std::make_shared<rng_t> (rng_t () );
	rng_ptr->seed(1234u);

	test<	multibeep::policies::random<num_t, rng_t> ,
			multibeep::bandits::empirical<num_t, rng_t> >(16, 5000, 5000, rng_ptr);

}

BOOST_AUTO_TEST_CASE(test_ucb){

	std::shared_ptr<rng_t> rng_ptr = std::make_shared<rng_t> (rng_t () );
	rng_ptr->seed(1234u);

	test<	multibeep::policies::UCB_p<num_t, rng_t>,
			multibeep::bandits::empirical<num_t, rng_t>
			> (128, 2048, 2048, rng_ptr,1);
}


BOOST_AUTO_TEST_CASE(test_prob_match){

	std::shared_ptr<rng_t> rng_ptr = std::make_shared<rng_t> (rng_t () );
	rng_ptr->seed(1234u);

	test<	multibeep::policies::prob_match<num_t, rng_t> ,
			multibeep::bandits::empirical<num_t, rng_t> > (16, 5000, 5000, rng_ptr);
}



BOOST_AUTO_TEST_CASE(test_succesive_halving){

	std::shared_ptr<rng_t> rng_ptr = std::make_shared<rng_t> (rng_t () );
	rng_ptr->seed(1234u);

	unsigned int num_arms = 128;
	unsigned int min_pulls = 16;
	num_t eta = 2;

	test<	multibeep::policies::successive_halving<num_t, rng_t>,
			multibeep::bandits::empirical<num_t, rng_t>
			> (num_arms, 8, 16384,min_pulls, eta);
}



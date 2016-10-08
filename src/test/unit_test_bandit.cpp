#include <random>

#include <boost/test/unit_test.hpp>

#include "multibeep/bandit/empirical_bandits.hpp"
#include "multibeep/bandit/posterior_bandit.hpp"

#include "multibeep/arm/exponential.hpp"
#include "multibeep/arm/bernoulli.hpp"
#include "multibeep/arm/normal.hpp"
//#include "multibeep/arm/data.hpp"
//#include "multibeep/arm/arm_generator_csv.hpp"


typedef std::default_random_engine rng_t;

//typedef boost::random::rand48 rng_t;

//typedef boost::random::mt19937 rng_t;
//typedef boost::random::knuth_b rng_t;

typedef double num_t;

template <typename bandit_t, typename ... T>
void basic_test (T ... t){
	std::shared_ptr<rng_t> rng_ptr = std::make_shared<rng_t> (rng_t () );
	rng_ptr->seed(1234u);
	
	bandit_t bandit(t...);
	auto i1 = bandit.add_arm(std::shared_ptr<multibeep::arms::base<num_t, rng_t> > (new multibeep::arms::normal_arm<num_t, rng_t> (0.5,0.2, rng_ptr)));
	
	BOOST_REQUIRE_EQUAL (i1, 0);
	BOOST_REQUIRE_EQUAL (bandit.number_of_active_arms(), 1);
	BOOST_REQUIRE_EQUAL (bandit.number_of_arms(), 1);
	BOOST_REQUIRE_EQUAL (bandit.number_of_pulls(), 0);
	BOOST_REQUIRE_EQUAL(bandit.number_of_pulled_arms(), 0);
	
	auto i2 = bandit.add_arm(std::shared_ptr<multibeep::arms::base<num_t, rng_t> > (new multibeep::arms::normal_arm<num_t, rng_t> (1.,0.3, rng_ptr)));
	
	BOOST_REQUIRE_EQUAL (i2, 1);
	BOOST_REQUIRE_EQUAL (bandit.number_of_active_arms(), 2);
	BOOST_REQUIRE_EQUAL (bandit.number_of_arms(), 2);
	BOOST_REQUIRE_EQUAL (bandit.number_of_pulls(), 0);
	BOOST_REQUIRE_EQUAL(bandit.number_of_pulled_arms(), 0);
	
	BOOST_REQUIRE_EQUAL(bandit[0].get_arm_ptr()->real_mean(), 0.5);
	
	BOOST_REQUIRE(std::isnan(bandit[0].estimated_mean));
	BOOST_REQUIRE(std::isnan(bandit[0].estimated_variance));
	
	
	double sum1;

	sum1 = bandit.pull_by_index(0);
	BOOST_REQUIRE_EQUAL(bandit.number_of_pulls(), 1);
	BOOST_REQUIRE_EQUAL(bandit[0].num_pulls, 1);
	BOOST_REQUIRE_EQUAL(bandit.number_of_pulled_arms(), 1);
	BOOST_REQUIRE_EQUAL(bandit[0].reward_stats.mean(), sum1);
	
	// no bandit should have a variance estimation after one pull
	BOOST_REQUIRE(std::isnan(bandit[0].estimated_variance));

	sum1 += bandit.pull_by_identifier(0);
	BOOST_REQUIRE_EQUAL(bandit.number_of_pulls(), 2);
	BOOST_REQUIRE_EQUAL(bandit.number_of_pulled_arms(), 1);
	BOOST_REQUIRE_EQUAL(bandit[0].num_pulls, 2);


	bandit.pull_by_identifier(1);
	BOOST_REQUIRE_EQUAL(bandit.number_of_pulls(), 3);
	BOOST_REQUIRE_EQUAL(bandit.number_of_pulled_arms(), 2);
	BOOST_REQUIRE_EQUAL(bandit[0].num_pulls, 2);
	BOOST_REQUIRE_EQUAL(bandit[1].num_pulls, 1);
	
	bandit.add_arm(std::shared_ptr<multibeep::arms::base<num_t, rng_t> > (new multibeep::arms::exponential_arm<num_t, rng_t> (0.1, rng_ptr)));
	bandit.add_arm(std::shared_ptr<multibeep::arms::base<num_t, rng_t> > (new multibeep::arms::bernoulli_arm<num_t, rng_t> (0.5, rng_ptr)));
	
	bandit.min_pull_arms(10);

    for (auto i=0u; i<bandit.number_of_active_arms(); i++){
        BOOST_REQUIRE(bandit[i].num_pulls >= 10);
	}

	for (auto i=0u; i<1000000; i++)
		bandit.pull_by_index(i%bandit.number_of_active_arms());

	bandit.update_p_max(false, 0.01, 64);
	bandit.deactivate_by_confidence_gap(0.01, false);
	
}



BOOST_AUTO_TEST_CASE(test_posterior_bandit){
	basic_test<multibeep::bandits::posterior<num_t,rng_t> >();
}


BOOST_AUTO_TEST_CASE(test_empirical_bandit){
	basic_test< multibeep::bandits::empirical<num_t, rng_t>	>();
}


BOOST_AUTO_TEST_CASE(test_n_last_pulls_bandit){
	basic_test< multibeep::bandits::last_n_pulls<num_t, rng_t>	>(5);
}


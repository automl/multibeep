#include <memory>
#include <random>
#include <cmath>

#include <boost/test/unit_test.hpp>


#include "multibeep/util/statistics.hpp"

#include "multibeep/arm/exponential.hpp"
#include "multibeep/arm/bernoulli.hpp"
#include "multibeep/arm/normal.hpp"



typedef std::default_random_engine rng_type;
typedef double num_t;


BOOST_AUTO_TEST_CASE(test_exponential_arm){
	std::shared_ptr<rng_type> rng_ptr = std::make_shared<rng_type> (rng_type () );
	int N = 1000000;
	double lambda = 23.5;
	
	multibeep::arms::exponential_arm<num_t, rng_type> arm (lambda, rng_ptr);
	
	multibeep::util::statistics::running_statistics<double> stat;
	
	
	BOOST_REQUIRE_CLOSE(arm.real_mean(), 1./lambda, 1e-6);
	BOOST_REQUIRE_CLOSE(arm.real_variance(), 1./(lambda*lambda),1e-6);
	
	
	for (auto i = 0; i < N; i++){
		auto r = arm.pull();
		BOOST_REQUIRE(r >= 0);
		stat(r);
	}

	BOOST_REQUIRE_CLOSE(stat.mean(), arm.real_mean(), 0.5);
	BOOST_REQUIRE_CLOSE(stat.variance(), arm.real_variance(), 0.5);
	
	auto p = arm.posterior();
	auto s = p->support(0.1);
	
	BOOST_REQUIRE_CLOSE(p->cdf(s.first), 0.05, 1e-5);
	BOOST_REQUIRE_CLOSE(p->cdf(s.second), 0.95, 1e-5);
	
	BOOST_REQUIRE_CLOSE(p->mean(), arm.real_mean(), 1e-1);
}


BOOST_AUTO_TEST_CASE(test_bernoulli_arm){
	std::shared_ptr<rng_type> rng_ptr = std::make_shared<rng_type> (rng_type () );
	int N = 10000000;
	double prob= 0.125;
	multibeep::util::statistics::running_statistics<double> stat;
	
	multibeep::arms::bernoulli_arm<float, rng_type> arm (prob, rng_ptr);
	
	BOOST_REQUIRE(arm.real_mean() == prob);
	BOOST_REQUIRE(arm.real_variance() == prob*(1-prob));
	
	for (auto i = 0; i < N; i++){
		auto r = arm.pull();
		BOOST_REQUIRE( ((r == 0) || (r == 1)));
		stat(r);
	}
	
	BOOST_REQUIRE_CLOSE(stat.mean(), arm.real_mean(), 0.5);
	BOOST_REQUIRE_CLOSE(stat.variance(), arm.real_variance(), 0.5);

	auto p = arm.posterior();
	auto s = p->support(0.1);
	
	BOOST_REQUIRE_CLOSE(p->cdf(s.first), 0.05, 1e-3);
	BOOST_REQUIRE_CLOSE(p->cdf(s.second), 0.95, 1e-3);
	
	
	BOOST_REQUIRE_CLOSE(p->mean(), arm.real_mean(), 1e-0);
	
	
	
}


BOOST_AUTO_TEST_CASE(test_normal_arm){
	std::shared_ptr<rng_type> rng_ptr = std::make_shared<rng_type> (rng_type () );
	int N = 1000000;
	double mu = 0.125;
	double sigma2 = 0.23;
	multibeep::util::statistics::running_statistics<double> stat;
	
	multibeep::arms::normal_arm<double, rng_type> arm(mu, sigma2, rng_ptr);
	
	BOOST_REQUIRE_CLOSE(arm.real_mean(), mu, 1e-10);
	BOOST_REQUIRE_CLOSE(arm.real_variance(), sigma2, 1e-10);
	
	for (auto i = 0; i < N; i++){
		auto r = arm.pull();
		stat(r);
	}
	
	BOOST_REQUIRE_CLOSE(stat.mean(), arm.real_mean(), 0.5);
	BOOST_REQUIRE_CLOSE(stat.variance(), arm.real_variance(), 0.5);

	auto p = arm.posterior();
	auto s = p->support(0.1);
	
	BOOST_REQUIRE_CLOSE(p->cdf(s.first), 0.05, 1e-6);
	BOOST_REQUIRE_CLOSE(p->cdf(s.second), 0.95, 1e-6);
	
	BOOST_REQUIRE_CLOSE(p->mean(), arm.real_mean(), 1e-0);
}


#ifndef MULTIBEEP_ARM_EXPONENTIAL
#define MULTIBEEP_ARM_EXPONENTIAL


#include <boost/random.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/math/distributions/inverse_gamma.hpp>

#include "multibeep/arm/arm.hpp"
#include "multibeep/util/statistics.hpp"

namespace multibeep{ namespace arms{


template<typename num_t = double, typename rng_t=std::default_random_engine>
class exponential_posterior: public multibeep::util::posteriors::boost_posterior<boost::math::inverse_gamma_distribution<num_t>, boost::random::uniform_real_distribution<num_t>, num_t, rng_t>{
	private:
		multibeep::util::statistics::running_statistics<num_t> stats;
		typedef multibeep::util::posteriors::boost_posterior<boost::math::inverse_gamma_distribution<num_t>, boost::random::uniform_real_distribution<num_t>, num_t, rng_t> base_t;
	public:
		exponential_posterior(multibeep::util::statistics::running_statistics<num_t> stat):
			stats(stat){
				base_t::predictive_posterior_dist = boost::random::uniform_real_distribution<num_t> (0,1);
				update_posteriors();
			}
		
		void update_posteriors (){
			base_t::posterior_dist = boost::math::inverse_gamma_distribution<num_t> (stats.number_of_points(), stats.number_of_points()*stats.mean());
		}
		
		virtual void add_observation (num_t v){
			stats(v);
			update_posteriors();
		} 
		virtual num_t predictive_posterior_sample (rng_t rng) const final{
			num_t u = base_t::predictive_posterior_dist(rng, base_t::predictive_posterior_dist.param());
			num_t alpha = stats.number_of_points();
			num_t lambda = stats.number_of_points()*stats.mean();
			
			return((std::pow((1-u), -1/alpha) - 1)*lambda);
		}
};


template<typename num_t = double, typename rng_t=std::default_random_engine>
class exponential_arm: public multibeep::arms::base<num_t, rng_t>{
	protected:
		std::shared_ptr<rng_t> rng_ptr;
		boost::random::exponential_distribution<num_t> rand_dist;
		multibeep::util::statistics::running_statistics<num_t> stats;
	public:  
		exponential_arm(num_t lambda, std::shared_ptr<rng_t> rp):
			rng_ptr(rp),
			rand_dist(boost::random::exponential_distribution<num_t>(lambda)) {}

		virtual num_t pull(){
			num_t reward = rand_dist(*rng_ptr);
			stats(reward);
			return(reward);
		}
		
		virtual double real_mean()		const	{return 1./(rand_dist.lambda());}
		virtual double real_variance()	const	{return(1./(rand_dist.lambda()*rand_dist.lambda()));	}
		virtual std::string get_ident()	const	{return "Exponential"; }
		virtual bool provides_posterior()	const {return(true);}


		
		virtual std::shared_ptr<typename multibeep::util::posteriors::base<num_t, rng_t> > posterior () const{
			try{
				return(std::shared_ptr<typename multibeep::util::posteriors::base<num_t, rng_t> > (
					new exponential_posterior<num_t, rng_t> (stats)
				));
			}
			catch (const std::domain_error &e){	return(std::shared_ptr<typename multibeep::util::posteriors::base<num_t, rng_t> > () );}
		}
};
}}
#endif

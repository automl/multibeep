#ifndef MULTIBEEP_ARM_NORMAL
#define MULTIBEEP_ARM_NORMAL

#include <cmath>

#include <boost/random.hpp>
//#include <boost/random/normal_distribution.hpp>
//#include <boost/random/student_t_distribution.hpp>
#include <boost/math/distributions/students_t.hpp>


#include "multibeep/arm/arm.hpp"
#include "multibeep/util/statistics.hpp"


namespace multibeep{ namespace arms{

// TODO: Prior for the variance seems to be to small!


template<typename num_t = double, typename rng_t=std::default_random_engine>
class normal_posterior: public multibeep::util::posteriors::scaled_boost_posterior<boost::math::students_t_distribution<num_t>, boost::random::student_t_distribution<num_t>, num_t, rng_t>{
	private:
		multibeep::util::statistics::running_statistics<num_t> stats;
		typedef multibeep::util::posteriors::scaled_boost_posterior<boost::math::students_t_distribution<num_t>, boost::random::student_t_distribution<num_t>, num_t, rng_t> base_t;
	public:
	
		virtual num_t center() const final {return(stats.mean());}
		virtual num_t scale()  const final {return( std::sqrt(stats.variance())/stats.number_of_points());}
		
	
		normal_posterior(multibeep::util::statistics::running_statistics<num_t> stat):
			base_t(boost::math::students_t_distribution<num_t> (stat.number_of_points()), boost::random::student_t_distribution<num_t> (stat.number_of_points())),
			stats(stat){}
		
		void update_posteriors (){
			base_t::posterior_dist =  boost::math::students_t_distribution<num_t> (stats.number_of_points());
			base_t::predictive_posterior_dist =  boost::random::student_t_distribution<num_t> (stats.number_of_points());
			
		}
		
		virtual void add_observation (num_t v){
			stats(v);
			update_posteriors();
		} 
		virtual num_t predictive_posterior_sample (rng_t &rng) const final{
			
			num_t x = base_t::predictive_posterior_dist(rng, base_t::predictive_posterior_dist.param());
			num_t n = stats.number_of_points();
			
			return(stats.mean() + std::sqrt(stats.variance()*(n+1))/n * x );
		}
};








	
template<typename num_t = double, typename rng_t=std::default_random_engine>
class normal_arm: public multibeep::arms::base<num_t, rng_t>{
	protected:
		std::shared_ptr<rng_t> rng_ptr;
		boost::random::normal_distribution<num_t> rand_dist;
		
		multibeep::util::statistics::running_statistics<num_t> stats;
		
	public:
		/* \brief creates arm from mean and standard deviation
		 * 
		 * \param mean the mean of the underlying Normal distribution
		 * \param variance the variance of it
		 */
		normal_arm(num_t mean, num_t variance, std::shared_ptr<rng_t> rp):
			rng_ptr(rp),
			rand_dist(boost::random::normal_distribution<num_t>(mean, std::sqrt(variance))) {}
		
		virtual num_t pull(){
			num_t reward = rand_dist(*rng_ptr);
			stats(reward);
			return(reward);
		}
		virtual double real_mean()		const	{return (rand_dist.mean());}
		virtual double real_variance()	const	{return (rand_dist.sigma()*rand_dist.sigma());}
		virtual std::string get_ident()	const	{return "Normal";};
		
		virtual bool provides_posterior() 	const	{return(true);}

		
		virtual std::shared_ptr<typename multibeep::util::posteriors::base<num_t, rng_t> > posterior () const{

			std::shared_ptr<typename multibeep::util::posteriors::base<num_t, rng_t> > ptr;
			// student t distribution requires 3 degrees of freedom for a variance
			// so we need 4 observations 
			if(stats.number_of_points() > 3){
				ptr = std::shared_ptr<typename multibeep::util::posteriors::base<num_t, rng_t> > (
					new normal_posterior<num_t, rng_t> (stats) );
			}
			else{
				ptr = std::shared_ptr<typename multibeep::util::posteriors::base<num_t, rng_t> > (NULL);
			}
			
			return(ptr);
		}
};
}}
#endif

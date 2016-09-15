#ifndef MULTIBEEP_UTIL_POSTERIOR
#define MULTIBEEP_UTIL_POSTERIOR

#include <random>
#include <boost/math/distributions.hpp>
#include <boost/random.hpp>

#include <multibeep/util/statistics.hpp>

namespace multibeep{ namespace util{ namespace posteriors{


/*brief unified interface for different posteriors*/
template <typename num_t = double, typename rng_t = std::default_random_engine>
class base{
	public:
		/* \brief the mean of the posterior over the mean reward*/
		virtual num_t mean() const = 0;
		/* \brief the variance of the posterior over the mean reward*/
		virtual num_t variance() const = 0;
		
		/*\brief PDF of the posterior at input*/
		virtual num_t pdf(num_t)	const = 0;
		/*\brief CDF of the posterior at input*/
		virtual num_t cdf(num_t)	const = 0;
		/*\brief returns the quantile of the input*/
		virtual num_t quantile(num_t)		const = 0;
		/*\brief support of the posterior 
		 * 
		 *  The returned interval contains 1-delta of the total weight. It is symmetrically
		 *  computed by CDF^(-1) (delta/2) and CDF^(-1) (1-delta/2).
		 * */
		virtual std::pair<num_t, num_t> support (num_t delta) const = 0;
		
		
		/* \brief can be used to sample from the predictive posterior*/
		virtual num_t predictive_posterior_sample (rng_t) const {
			throw std::runtime_error("This posterior does not support sampling from the predictive posterior!");
		}
		
		/* \brief updates the posterior based on the input, e.g., for predicting change in entropy */
		virtual void add_observation (num_t){
			throw std::runtime_error("This posterior does not support adding an observation!");
		}
};


/*brief posterior based on any simple boost math distribution*/
template <typename boost_math_distribution, typename boost_rand_distribution, typename num_t = double,  typename rng_t = std::default_random_engine>
class boost_posterior: public base<num_t, rng_t>{
	protected:
		boost_math_distribution posterior_dist;
		boost_rand_distribution predictive_posterior_dist;
	public:

		virtual num_t mean()		const final {
			try{return(boost::math::mean(posterior_dist));}
			catch (const std::domain_error &e){return(NAN);}
		}
		virtual num_t variance()	const final {
			try{ return(boost::math::variance(posterior_dist));}
			catch (const std::domain_error &e){return(NAN);}
		}
		virtual num_t pdf(num_t x)	const final {
			try{ return(boost::math::pdf(posterior_dist,x)); }
			catch (const std::domain_error &e){return(NAN); }
		}
		virtual num_t cdf(num_t x)	const final {
			try{ return(boost::math::cdf(posterior_dist,x));}
			catch (const std::domain_error &e){return(NAN);}
		}
		virtual num_t quantile(num_t x)	const final {
			try{ return(boost::math::quantile(posterior_dist,x));}
			catch (const std::domain_error &e){return(NAN);}
		}
		virtual std::pair<num_t, num_t> support (num_t delta) const final {
			try{
				if ((delta  <= 0) || (delta >= 1))
					return(boost::math::support(posterior_dist));
				delta = std::min(delta, 1-delta);
				return(std::pair<num_t, num_t> (boost::math::quantile(posterior_dist, delta/num_t(2.)), boost::math::quantile(posterior_dist, 1-delta/num_t(2))));
			}
			catch (const std::domain_error &e){	return(std::pair<num_t,num_t> (NAN,NAN));}
		}	
		
		virtual num_t predictive_posterior_sample (rng_t rng) const {return(predictive_posterior_dist(rng, predictive_posterior_dist.param()));}
};


/*brief posterior based on a scaled version of any boost math distribution*/
template <typename boost_math_distribution, typename boost_rand_distribution, typename num_t = double,  typename rng_t = std::default_random_engine>
class scaled_boost_posterior: public base<num_t, rng_t>{
	protected:
		boost_math_distribution posterior_dist;
		boost_rand_distribution predictive_posterior_dist;

	public:

		virtual num_t center () const = 0;
		virtual num_t scale  () const = 0;

		scaled_boost_posterior() = default;
		
		scaled_boost_posterior(boost_math_distribution dist1, boost_rand_distribution dist2):
			posterior_dist(dist1), predictive_posterior_dist(dist2) {}
		
	
		num_t scale_x	(num_t x) const {return( (x - center()) / scale() );}
		num_t descale_x	(num_t x) const {return( (x * scale() ) + center());}
		
		virtual num_t mean()		const final {
			try{return(boost::math::mean(posterior_dist) + center());}
			catch (const std::domain_error &e){	return(NAN); }
		}
		virtual num_t variance()	const final {
			try{return(boost::math::variance(posterior_dist)*scale()*scale());}
			catch (const std::domain_error &e){	return(NAN); }
		}
		virtual num_t pdf(num_t x)	const final {
			try{return(boost::math::pdf(posterior_dist,scale_x(x))/scale());}
			catch (const std::domain_error &e){	return(NAN); }
		}
		virtual num_t cdf(num_t x)	const final {
			try{return(boost::math::cdf(posterior_dist,scale_x(x)));}
			catch (const std::domain_error &e){	return(NAN); }
		}
		virtual num_t quantile(num_t x)	const final {
			try{ return(descale_x(boost::math::quantile(posterior_dist,x)));}
			catch (const std::domain_error &e){	return(NAN);}
		}
		virtual std::pair<num_t, num_t> support (num_t delta) const final {
			std::pair<num_t, num_t> rv;
			try{
				if ((delta  <= 0) || (delta >= 1))
					rv = boost::math::support(posterior_dist);
				
				else{
					delta = std::min(delta, 1-delta);
					rv.first  = boost::math::quantile(posterior_dist,    delta/num_t(2));
					rv.second = boost::math::quantile(posterior_dist, 1- delta/num_t(2));
				}
				
				rv.first = descale_x(rv.first);
				rv.second= descale_x(rv.second);
			}
			catch (const std::domain_error &e){
				rv.first = NAN; rv.second = NAN;
			}
			return(rv);
		}
				
		virtual num_t predictive_posterior_sample (rng_t rng) const {return(predictive_posterior_dist(rng, predictive_posterior_dist.param()));}


};



/*\brief posterior without any prediction based on a boost::math distribution */
template <typename boost_math_distribution_t, typename num_t = double,  typename rng_t = std::default_random_engine>
class simple_posterior: public base<num_t, rng_t>{
	protected:
		boost_math_distribution_t posterior_dist;
	
	public:
		template <typename ... A>
		simple_posterior (A ... args): posterior_dist(args...) {}
	
		virtual num_t mean()		const final {
			try{return(boost::math::mean(posterior_dist));}
			catch (const std::domain_error &e){return(NAN);}
		}
		virtual num_t variance()	const final {
			try{ return(boost::math::variance(posterior_dist));}
			catch (const std::domain_error &e){return(NAN);}
		}
		virtual num_t pdf(num_t x)	const final {
			try{ return(boost::math::pdf(posterior_dist,x)); }
			catch (const std::domain_error &e){return(NAN); }
		}
		virtual num_t cdf(num_t x)	const final {
			try{ return(boost::math::cdf(posterior_dist,x));}
			catch (const std::domain_error &e){return(NAN);}
		}
		virtual num_t quantile(num_t x)	const final {
			try{ return(boost::math::quantile(posterior_dist,x));}
			catch (const std::domain_error &e){return(NAN);}
		}
		virtual std::pair<num_t, num_t> support (num_t delta) const final {
			try{
				if ((delta  <= 0) || (delta >= 1))
					return(boost::math::support(posterior_dist));
				delta = std::min(delta, 1-delta);
				return(std::pair<num_t, num_t> (boost::math::quantile(posterior_dist, delta/num_t(2.)), boost::math::quantile(posterior_dist, 1-delta/num_t(2))));
			}
			catch (const std::domain_error &e){	return(std::pair<num_t,num_t> (NAN,NAN));}
		}
	
};

template <typename num_t = double,  typename rng_t = std::default_random_engine>
class gaussian_posterior: public simple_posterior< boost::math::normal_distribution<num_t>, num_t, rng_t>{
	protected:
		typedef simple_posterior< boost::math::normal_distribution<num_t>, num_t, rng_t> base_t;
	public:
		gaussian_posterior (num_t mean, num_t variance): base_t(mean, std::sqrt(variance)) {}
};



}}}
#endif

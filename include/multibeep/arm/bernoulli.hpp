#ifndef MULTIBEEP_ARM_BERNOULLI
#define MULTIBEEP_ARM_BERNOULLI

#include <boost/random.hpp>
#include <boost/random/bernoulli_distribution.hpp>
#include <boost/math/distributions/beta.hpp>
#include <multibeep/arm/arm.hpp>

namespace multibeep{ namespace arms{

template<typename num_t = double, typename rng_t=std::default_random_engine>
class bernoulli_arm: public multibeep::arms::base< num_t, rng_t >{
	protected:
		std::shared_ptr<rng_t> rng_ptr;
		boost::random::bernoulli_distribution<num_t> rand_dist;

		unsigned long long int N0, N1;

	public:  
		bernoulli_arm(num_t p, std::shared_ptr<rng_t> rp): rng_ptr(rp),
			rand_dist(boost::random::bernoulli_distribution<num_t>(p)),
			N0(0), N1(0) {};
		
		
		virtual num_t pull(){
			num_t res = rand_dist(*rng_ptr);
			if (res < 0.5) N0++;
			else N1++;
			return(res);
		}
		virtual num_t real_mean()		const	{return(rand_dist.p());	}
		virtual num_t real_variance()	const	{return(rand_dist.p() * (1-rand_dist.p()));}
		virtual std::string get_ident()	const	{return("Bernoulli");}
		
		virtual bool provides_posterior()	const	final {return(true);}


		class bernoulli_posterior: public multibeep::util::posteriors::boost_posterior<boost::math::beta_distribution<num_t>, boost::random::bernoulli_distribution<num_t>, num_t, rng_t>{
			private:
				unsigned long long int N0;
				unsigned long long int N1;
				typedef multibeep::util::posteriors::boost_posterior<boost::math::beta_distribution<num_t>, boost::random::bernoulli_distribution<num_t>, num_t, rng_t> base_t;
			public:
				bernoulli_posterior(unsigned long long int n0, unsigned long long int n1): N0(n0), N1(n1) {update_posteriors();}
				
				void update_posteriors (){
					base_t::posterior_dist = boost::math::beta_distribution<num_t> (N1+1, N0+1);
					base_t::predictive_posterior_dist = boost::random::bernoulli_distribution<num_t>((N1+1)/(N0+N1+2));
				}
				
				virtual void add_observation (num_t v){
					if (v<0.5) ++N0;
					else ++N1;
					update_posteriors();
				} 
		};


		virtual std::shared_ptr<typename multibeep::util::posteriors::base<num_t, rng_t> > posterior () const{
			return (
				std::shared_ptr<typename multibeep::util::posteriors::base<num_t, rng_t> > (
					new bernoulli_posterior(N0,N1)
				)
			);
		}










	};
}}
#endif

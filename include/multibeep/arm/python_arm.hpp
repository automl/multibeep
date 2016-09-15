#ifndef MULTIBEEP_ARM_PYTHONARM
#define MULTIBEEP_ARM_PYTHONARM

#include <string>
#include "arm.hpp"
#include <multibeep/util/posteriors.hpp>

namespace multibeep{ namespace arms{

	template <typename num_t = double, typename rng_t = std::default_random_engine>
	class python_arm: public base<num_t>{
		protected:
			
			// wrapper for pulling
			typedef num_t (*python_pull_wrapper)(void*);
			
			// wrapper for the posterior
			typedef std::shared_ptr<multibeep::util::posteriors::base<num_t, rng_t> > (*python_posterior_wrapper)(void*);
			
			typedef void (*python_deactivate_wrapper)(void*);
			
			std::string name;
			void * python_object;
			python_pull_wrapper python_pull;
			python_pull_wrapper python_mean;		// I know this is kind of bad style to recycle the typedef python_pull_wrapper
			python_pull_wrapper python_variance;	// here, but adding more types seems a bit too much
			python_posterior_wrapper python_posterior;
			python_deactivate_wrapper python_deactivate;
		
		public:
			python_arm( std::string n, void * po,
						python_pull_wrapper p_pull,
						python_pull_wrapper p_mean,
						python_pull_wrapper p_variance,
						python_posterior_wrapper p_post,
						python_deactivate_wrapper p_deactivate
						):
				name(n), python_object(po), python_pull(p_pull),
				python_mean(p_mean), python_variance(p_variance),
				python_posterior(p_post), python_deactivate(p_deactivate) {}
		
			virtual num_t pull(){ return(python_pull(python_object));}
			/*\brief known real mean of the arm to compute regrets*/
			virtual num_t real_mean() const {return(python_mean(python_object));};
			/*\brief known variance of the arm; not really necessary*/
			virtual num_t real_variance() const {return(python_variance(python_object));};

			virtual std::string get_ident() const {return(name);};
			
			virtual bool provides_posterior()	const	final {return(true);}
			
			virtual std::shared_ptr<typename multibeep::util::posteriors::base<num_t, rng_t> > posterior () const
				{return(python_posterior(python_object));}
			
			virtual void deactivate(){ python_deactivate(python_object);}
			
	};
}}
#endif

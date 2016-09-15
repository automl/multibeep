#ifndef MULTIBEEP_ARM
#define MULTIBEEP_ARM

#include <stdexcept>
#include <string>
#include <memory>
#include <multibeep/util/posteriors.hpp>
#include <multibeep/util/reward_predictor.hpp>

namespace multibeep{ namespace arms{

	template <typename num_t = double, typename rng_t = std::default_random_engine>
	class base{
		public:
			/*\brief pulls arm and returns reward */
			virtual num_t pull() = 0;
			/*\brief known real mean of the arm to compute regrets*/
			virtual num_t real_mean() const = 0;
			/*\brief known variance of the arm; not really necessary*/
			virtual num_t real_variance() const = 0;
			/*\brief every arm type has a unique id; for storage */
			virtual std::string get_ident() const = 0;
			
			virtual bool provides_posterior() const { return(false);}
			
			virtual std::shared_ptr<multibeep::util::posteriors::base<num_t, rng_t> > posterior() const {
				//TODO: insert ident into the string for better error messages
				throw std::runtime_error("Arm does not provide a posterior!");
			}
			
			/* \brief function allowing some tear-down when the arm is deactivated*/
			virtual void deactivate () {}
			
			virtual ~base() {}
	};
	
}}
#endif

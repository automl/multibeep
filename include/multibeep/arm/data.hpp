#ifndef MULTIBEEP_ARM_DATA
#define MULTIBEEP_ARM_DATA
#include <algorithm>
#include <random>

#include "multibeep/arm/arm.hpp"
#include "multibeep/util/statistics.hpp"

namespace multibeep{ namespace arms{
	template< bool bootstrap, typename num_t = double, typename rng_t = std::default_random_engine>
	class data_arm: public base<num_t>{	

		std::shared_ptr<rng_t> rng_ptr;
		std::uniform_int_distribution<unsigned int> u;

		num_t real_mean_ = NAN;
		num_t real_variance_ = NAN;
		unsigned int idx = std::numeric_limits<unsigned int>::max();
		
		std::vector<num_t> data;
		std::string name;

	public:
		data_arm(	num_t * values, unsigned int num_values, std::string name,
					std::shared_ptr<rng_t> r_ptr): rng_ptr(r_ptr),name(name){
						
			u = std::uniform_int_distribution<unsigned int> (0, num_values-1);
			data.reserve(num_values);
			
			multibeep::util::statistics::running_statistics<num_t> stat;
			
			for (auto i=0u; i < num_values; i++){
				stat(values[i]);
				data.push_back(values[i]);
			}

			real_mean_ = stat.mean();
			real_variance_ = stat.variance();
		}
	
		
		data_arm(	std::vector<double>& data,	std::string name,
					std::shared_ptr<rng_t> r_ptr): data_arm( data.data(), data.size(), name, r_ptr){}
		
		virtual num_t pull(){
			num_t v;
			if (bootstrap){
				idx = u(*rng_ptr);

			}else{
				idx = (idx+1)%data.size();
			}
			v = data.at(idx);
			return v;
		};
		virtual num_t real_mean()		const	{return(real_mean_);}
		virtual num_t real_variance()	const	{return(real_variance_);}

		virtual bool provides_posterior()	const { return(false);}

		std::string get_ident() const { return std::string("Data") + std::string(" ") + name;}
		
	};
	


	/* \brief redundant class in C++, but Python interface struggles with booleans as template arguments */ 
	template<typename num_t = double, typename rng_t = std::default_random_engine>
	class data_arm_bootstrap: public data_arm<true, num_t, rng_t>{

		typedef data_arm<true, num_t, rng_t> base_t;

		public:
		
			data_arm_bootstrap(	std::vector<double>& data,	std::string name, std::shared_ptr<rng_t> r_ptr):
				base_t( data, name, r_ptr){}

			data_arm_bootstrap(	num_t * data, unsigned int num_values, std::string name, std::shared_ptr<rng_t> r_ptr):
				base_t(data, num_values, name, r_ptr){}
		
	};

	/* \brief redundant class in C++, but Python interface struggles with booleans as template arguments */ 
	template<typename num_t = double, typename rng_t = std::default_random_engine>
	class data_arm_sequential: public data_arm<false, num_t, rng_t>{

		typedef data_arm<false, num_t, rng_t> base_t;

		public:
		
			data_arm_sequential(	std::vector<double>& data,	std::string name, std::shared_ptr<rng_t> r_ptr):
				base_t( data, name, r_ptr){}

			data_arm_sequential(	num_t * data, unsigned int num_values, std::string name, std::shared_ptr<rng_t> r_ptr):
				base_t(data, num_values, name, r_ptr){}
		
	};




}}
#endif

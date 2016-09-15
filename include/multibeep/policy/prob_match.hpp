#ifndef MULTIBEEP_POLICY_GAUSS_MATCH
#define MULTIBEEP_POLICY_GAUSS_MATCH

#include "multibeep/policy/policy.hpp"
#include "multibeep/bandit/bandit.hpp"

namespace multibeep{ namespace policies{

	template<typename num_t=double, typename rng_t = std::default_random_engine>
	class prob_match: public multibeep::policies::base<num_t, rng_t>{
		protected:
			typedef multibeep::policies::base<num_t, rng_t> base_t;
			std::shared_ptr<rng_t> rng_ptr;
			
		public:
			prob_match(std::shared_ptr<multibeep::bandits::base<num_t, rng_t> > b_ptr, std::shared_ptr<rng_t> r_ptr):
				base_t(b_ptr), rng_ptr(r_ptr) {}
			
			std::string get_ident() {return(std::string("prob_match"));}
			
			virtual unsigned int select_next_arm(){
				// convenience alias for the bandit
				auto &b (*(base_t::bandit_ptr));
				
				std::uniform_real_distribution<num_t> u(0,1);

				
				num_t max = std::numeric_limits<num_t>::lowest();
				unsigned int index = 0;
				
				// loop through the rest
				for (auto i=0u; i <  b.number_of_active_arms(); i++){
					// pull arms that have no propper posterior yet
					if (!b[i].posterior)	return(i);
					// draw a random mean from the posterior
					num_t sample(b[i].posterior->quantile( u(*rng_ptr) ) );
					// pull arms where the quantile computation returned NAN should
					// only happen if there was a domain_error, i.e. usually not enough pulls
					if (std::isnan(sample)) return(i);
					// store index and value of maximum
					if (sample > max){
						max = sample;
						index = i;
					}
				}
				return(index);
			}
	};
/*	template<class BanditClass, class Engine>
	class PolicyGaussMatch: public Policy<BanditClass>{
	protected:
		Engine& rng;
		boost::random::uniform_int_distribution<unsigned int>u;
		unsigned int N_samples;
	public:
		PolicyGaussMatch(BanditClass& bandit, Engine& rng, unsigned int N_samples): Policy<BanditClass>(bandit),rng(rng), N_samples(N_samples){
			u = boost::random::uniform_int_distribution<unsigned int>(0,this->bandit.num_active_arms()-1);
		};
		std::string  get_ident(){
			return std::string("PolicyGaussMatch");
		}
		const ArmInfo& select_next_arm(){
			std::pair<bool, unsigned int> hast_to_pull = this->pull_at_least(this->num_pull_at_least, rng);
			if (hast_to_pull.first){
				return this->bandit.get_arm_infos()[hast_to_pull.second];
			}
			std::vector<double> pmax = pmax::get_current_pmax_normal(this->bandit, false, this->N_samples);
			std::discrete_distribution<int> distribution(pmax.begin(), pmax.end());
			return this->bandit.get_arm_infos()[distribution(rng)];
			
		};
	};
	*/
}}
#endif

#ifndef MULTIBEEP_EMPIRICAL_BANDITS
#define MULTIBEEP_EMPIRICAL_BANDITS

#include "multibeep/bandit/bandit.hpp"




namespace multibeep{ namespace bandits{



template <typename num_t = double, typename rng_t = std::default_random_engine>
class empirical: public base<num_t, rng_t>{
	
	typedef base<num_t,rng_t> base_t;
	
	public:
		virtual void update_arm_info(unsigned int index){
			auto &ai = base_t::arm_infos.at(index);
			if (ai.dirty){
				// empirical stats require at least 2 pulls to make sense :)
				if (ai.reward_stats.number_of_points() >1) {
					ai.estimated_mean = ai.reward_stats.mean();
					ai.estimated_variance = std::max(1e-6, ai.reward_stats.variance()/ai.reward_stats.number_of_points());

					ai.posterior =
						std::shared_ptr<multibeep::util::posteriors::base<num_t, rng_t> > (
							new multibeep::util::posteriors::gaussian_posterior<num_t, rng_t> (
								ai.reward_stats.mean(),
								std::max(std::numeric_limits<num_t>::min(), ai.reward_stats.variance()/ai.reward_stats.number_of_points())
							)
					);
				}
				else 
					ai.posterior = std::shared_ptr<multibeep::util::posteriors::base<num_t, rng_t> > (NULL);
				
				ai.dirty=false;
				base_t::num_dirty_arms--;
			}
		}
};



template <typename num_t = double, typename rng_t = std::default_random_engine>
class last_n_pulls: public base<num_t, rng_t>{
	
	typedef base<num_t, rng_t> base_t;
	unsigned int n;
	
	
	public:
	
		last_n_pulls(unsigned int N): base_t(), n(N) {}
	
		virtual void update_arm_info(unsigned int index){
			auto &ai = base_t::arm_infos.at(index);

			if (ai.dirty){
				// compute empirical statistics from last n rewards
				multibeep::util::statistics::running_statistics<num_t> stats;
				for (auto it=ai.rewards.rbegin(); it!=ai.rewards.rend(); it++){
					stats(*it);
					if (stats.number_of_points() == n) break;
				}

				ai.estimated_mean = stats.mean();
				ai.estimated_variance = stats.variance();

				if ( isnan(stats.variance()))
					ai.posterior = std::shared_ptr<multibeep::util::posteriors::base<num_t, rng_t> > (NULL);
				else
					ai.posterior =
						std::shared_ptr<multibeep::util::posteriors::base<num_t, rng_t> > (
							new multibeep::util::posteriors::gaussian_posterior<num_t, rng_t> (
								stats.mean(),
								std::max(std::numeric_limits<num_t>::min(), stats.variance()/stats.number_of_points())
							)
							
					);
				ai.dirty=false;
			}
		}
};


//template <typename trajectory_t = multibeep::bandits::minimal_bandit_trajectory<empirical_arm_info> >
//class empirical_bandit: public multibeep::bandits::proto_bandit<empirical_arm_info, trajectory_t> {};


/* Joels old code for his robust empirical bandit
 * TODO: refactor that into the new class hierarchy

class empirical_bandit: public bandit{
		
	public:
		//double alpha = 5.0;
		
		Bandit(): K(0), N(0), q(0), sum_reward(0){
			arm_infos = std::vector<ArmInfo>();
		};
		virtual unsigned int push_back(Arm* arm){
			++K;
			unsigned int i = arm_infos.size();
			arm_infos.emplace_back(arm, i);
			active_arms_idx.push_back(i);
			const ArmInfo& ai = arm_infos[i];
			for (unsigned npulls = 0; npulls < min_pulls; ++npulls){
				pull(ai);
			}
			return i;
		};
		virtual unsigned int get_min_pulls(){
			return min_pulls;
		}
		virtual void deactivate(unsigned int pos){
			if (arm_infos[pos].is_active){
				--K;
				for (auto it = std::begin(active_arms_idx); it!=std::end(active_arms_idx); ++it){
					unsigned int active_pos = *it;
					if (active_pos == pos){
						active_arms_idx.erase(it);
						break;
					}
				}
				ArmInfo& ai = arm_infos[pos] ;
				ai.is_active= false;
			}
		};
		virtual void activate(unsigned int pos){
			if (!arm_infos[pos].is_active){
				++K;
				arm_infos[pos].is_active = true;
				active_arms_idx.push_back(pos);
			}
		};
		virtual void deactivate(BanditIterator& it){
			unsigned int active_idx = get_active_arms_idx()[it.p];
			deactivate(active_idx);
		}
		virtual void activate(BanditIterator& it){
			unsigned int active_idx = get_active_arms_idx()[it.p];
			activate(active_idx);
		}
		virtual const std::vector<ArmInfo>& get_arm_infos(){
			return this->arm_infos;
		};
		virtual const std::vector<unsigned int>& get_active_arms_idx(){
			return active_arms_idx;
		};
		virtual unsigned int num_pulled(){
			return N;
		}
		virtual const ArmInfo& get_best_arm_by_mean(){
			auto it = this->begin();
			const ArmInfo* best_arm_info = &(*(it));
			
			double best_mean;
			if (this->is_model_used()){
				best_mean = best_arm_info->estimated_mean;
			}else{
				best_mean =  best_arm_info->sample_mean;
			}
			++it;
			for (; it !=this->end(); ++it){
				double this_mean;
				if (this->is_model_used()){
					this_mean = (*(it)).estimated_mean;
				}else{
					this_mean = (*(it)).sample_mean;
				}
				if (this_mean > best_mean && (*(it)).number_pulled > 0){
					best_arm_info =  &(*(it));
					best_mean = this_mean;
				}
			}
			return *best_arm_info;
		}
		virtual std::vector<ArmInfo> get_sorted_arm_infos_by_mean(){
			std::vector<ArmInfo> return_vector = this->arm_infos;
			if (this->is_model_used()){
				std::sort(return_vector.begin(), return_vector.end(), ArmInfo::is_estimated_mean_bigger);
			}
			else{
				std::sort(return_vector.begin(), return_vector.end(), ArmInfo::is_sample_mean_bigger);
			}
			return return_vector;
		}
		virtual unsigned int num_active_arms(){
			return K;
		};
		virtual unsigned int num_arms(){
			return this->arm_infos.size();
		};
		virtual void pull(const ArmInfo& arm_info){
			assert( 1 ==  arm_info.is_active);
			if (arm_info.number_pulled == 0){
				++q;
			}
			double reward = arm_info.arm->pull();
			this->update(arm_info, reward);
		};
		virtual void pull(const ArmInfo& arm_info, int instance_idx){
			if (instance_idx == -1){
				pull(arm_info);
				return;
			}
			assert( 1 ==  arm_info.is_active);
			if (arm_info.number_pulled == 0){
				++q;
			}
			double reward = arm_info.arm->pull(instance_idx);
			this->update(arm_info, reward);
		};
		virtual void pull(unsigned int ident){
			this->pull(arm_infos[ident]);
		};
		
		virtual void pull(unsigned int ident, int instance_idx){
			this->pull(arm_infos[ident], instance_idx);
		};
		
		virtual unsigned int num_arms_pulled(){
			return q;
		}

		virtual const std::vector<BanditTrajectory>& get_trajectory(){
			return trajectory;
		}
		
		virtual void reset(long long unsigned int){
			N=0;
			K=0;
			sum_reward = 0;
			q= 0;
			mean_mean = 0;
			mean_of_std_dev = 1.00;
			mean_of_variance = 1.00;
			std_dev_of_mean = 1.00;
			total_variance = 1.00;
			std::vector<ArmInfo> arm_infos_tmp = arm_infos;
			this->arm_infos = std::vector<ArmInfo>();//this->num_arms());
			this->active_arms_idx = std::vector<unsigned int>();
			this->trajectory = std::vector<BanditTrajectory>(); 
			for (unsigned int i = 0; i < arm_infos_tmp.size(); ++i){
				this->push_back(arm_infos_tmp[i].arm);
			}
		}

		virtual void update(const ArmInfo& arm_info, double reward) = 0;		
		virtual void pseudo_update(const ArmInfo&, double){
			throw std::runtime_error("Pseudo Update not supported!");
		}
		virtual void undo_pseudo_update(){
			throw std::runtime_error("Undo Pseudo Update not supported!");
		}
		void update_arm_info(ArmInfo* arm_info_ptr, double reward){
			arm_info_ptr->sum_reward += reward;
			arm_info_ptr->trajectory.emplace_back(N, reward, sum_reward, arm_info_ptr->ident);
			arm_info_ptr->sum_squared_reward +=reward*reward;
			++(arm_info_ptr->number_pulled);
			arm_info_ptr->sample_mean = arm_info_ptr->sum_reward/arm_info_ptr->number_pulled;
			if (arm_info_ptr->number_pulled > 1){
				arm_info_ptr->sample_variance = 1.0/(arm_info_ptr->number_pulled-1)*(
					arm_info_ptr->sum_squared_reward - 
					(arm_info_ptr->sum_reward* arm_info_ptr->sum_reward)/
					(arm_info_ptr->number_pulled));
			}
			else{
				arm_info_ptr->sample_variance = this->total_variance;
			}
			arm_info_ptr->robust_variance = this->get_robust_variance_prediction(arm_info_ptr);
			if (isnan(arm_info_ptr->sample_variance) || arm_info_ptr->sample_variance < 0){
				arm_info_ptr->sample_variance = 0.000000001;
			}
		}
		// updates mean_mean and mean_of_std_dev. Should be called after updateing estimated variance 
		//and mean of the corresponding arm
		virtual void update_globals(){
			double sum_mean = 0.0;
			double sum_squared_mean = 0.0;
			double sum_std_dev = 0.0;
			double sum_variance = 0.0;
			unsigned int known_mean = 0;
			unsigned int known_std_dev = 0;
			for (auto it = begin(); it != end(); ++it){
				if ((*it).number_pulled > 0){
					sum_mean += (*it).sample_mean;
					sum_squared_mean += (*it).sample_mean*(*it).sample_mean;
					++known_mean;
				}
				if  ((*it).number_pulled > 1){
					sum_std_dev += std::sqrt((*it).sample_variance);
					sum_variance += (*it).sample_variance;
					++known_std_dev;
				}
			}
			if (known_std_dev == 0 || sum_std_dev == 0){
				this->mean_of_std_dev = 1.0;
			}else{
				this->mean_of_std_dev = sum_std_dev / known_std_dev;
				this->mean_of_variance = sum_variance / known_std_dev;
			}
			this->mean_mean = sum_mean / known_mean;
			this->known_mean = known_mean;
			this->known_std_dev = known_std_dev;
			double variance_of_mean = sum_squared_mean/known_mean  - (mean_mean*mean_mean);
			this->std_dev_of_mean = std::sqrt(variance_of_mean);
			
			this->total_variance = this->mean_of_variance +  variance_of_mean;
			//std::cout << "updated to " << this->total_variance << " = " << this->mean_of_variance <<" + "<<  variance_of_mean<<"\n";
		};
		//drawing variance from an inverse gamma distribution
		auto get_robust_variance_prediction(const ArmInfo* arm_info) -> double{
			if (total_variance <= 0){
				return 1.0;
			}
			if (alpha < 0.000001){
				return  arm_info->sample_variance + 0.000001;
				
			}
			double n = arm_info->number_pulled ;
			if (n > 0){
				n = 0.5*(n-1);
			}
			double sigma_square = arm_info->sample_variance;
			
			double h1 = n +alpha;
			boost::math::inverse_gamma_distribution<double> inv_gamma(h1, n*sigma_square + total_variance*alpha);
			
			double v = quantile(inv_gamma, 0.8);
			return v;
		}
		virtual double get_total_variance(){
			return total_variance;
		}
		virtual double get_global_mean(){
			return mean_mean;
		}
		virtual double get_mean_of_std_dev(){
			return mean_of_std_dev;
		}
		virtual double get_mean_of_variance(){
			return mean_of_variance;
		}

		virtual double get_std_dev_of_mean(){
			return std_dev_of_mean;
		}
		virtual double get_estimated_N() = 0;
		virtual unsigned int get_num_instances(){
			return 0;
		}
		virtual void update_estimated_values(){};
		bool is_model_used(){
			return model_used;
		}
		virtual std::vector<double> sample_rewards_from_model(const ArmInfo& , std::vector<unsigned int> ){
			std::vector<double> rewards;
			return rewards;
		}
	};
*/

}} // namespaces
#endif

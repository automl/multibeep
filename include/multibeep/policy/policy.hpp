#ifndef MULTIBEEP_POLICY
#define MULTIBEEP_POLICY

#include <string>
#include <memory>
#include <random>

#include "multibeep/bandit/bandit.hpp"



namespace multibeep{ namespace policies{
	
	
	template<typename num_t = double, typename rng_t = std::default_random_engine>
	class base{
		
		protected:
			std::shared_ptr<multibeep::bandits::base<num_t, rng_t> > bandit_ptr;
		public:

			base (std::shared_ptr<multibeep::bandits::base<num_t,rng_t> > b_ptr):
				bandit_ptr(b_ptr) {}
		
			/* \brief returns the next arm to pull based on the */
			virtual unsigned int select_next_arm() = 0;
		
			/* \brief selects and pulls the arm num_rounds times */
			virtual void play_n_rounds (unsigned int num_rounds){
				while (num_rounds > 0){
					bandit_ptr->pull_by_index(select_next_arm());
					--num_rounds;
				}
			}
			
			virtual std::string  get_ident() = 0;	
			virtual ~base() {}
	};




/*

	template<class BanditClass>
	class Policy{
	protected:
		BanditClass& bandit;
		unsigned int num_pull_at_least = 2;
	public:
		Policy(BanditClass& bandit):bandit(bandit){};
		virtual const ArmInfo& select_next_arm() = 0;
		virtual const std::pair<ArmInfo, int> select_arm_and_instance(){
			std::pair<ArmInfo, int> arm_and_instance(this->select_next_arm(), -1);
			return arm_and_instance;
		};
		virtual std::string get_ident() = 0;
		
		virtual BanditClass& get_bandit(){
			return bandit;
		}
		void set_num_pull_at_least(unsigned int num_pull_at_least){
			this->num_pull_at_least = num_pull_at_least;
		}
		unsigned int get_num_pull_at_least(){
			return this->num_pull_at_least;
		}
		template<class V,class RNG>
		unsigned int select_max_idx(std::vector<V>& values, std::vector<unsigned int>& idx, RNG& rng){
			double best_reward = values[0];
			std::vector<unsigned int> best_reward_idx = {idx[0]};
			for(unsigned int i = 1; i < values.size(); ++i){
				double reward = values[i];
				if (reward > best_reward){
					best_reward = reward;
					best_reward_idx.assign(1,idx[i]);
				}
				else if(reward == best_reward){
					best_reward_idx.emplace_back(idx[i]);
				}
			}
			boost::random::uniform_int_distribution<unsigned int>u(0,best_reward_idx.size()-1);
			return best_reward_idx[u(rng)];
		}
		template<class V,class RNG>
		unsigned int select_max_idx(std::vector<V>& values_all, RNG& rng){
			std::vector<double> values;
			std::vector<unsigned int> idx;
			for (auto it = this->bandit.begin(); it != this->bandit.end(); ++it){
				double v = values_all[(*it).ident];
				values.push_back(v);
				idx.push_back((*it).ident);
			}
			return select_max_idx(values, idx, rng);
		}
		 //reorder and fill the vector so it has the same indizes like all bandit->get_arm_infos
		template<class T>
		std::vector<T> reorder_to_complete_arm_set(std::vector<T>& active_only, T fill_value){
			std::vector<T> result(this->bandit.num_arms(), fill_value);
			std::vector<unsigned int> active_arm_idx = this->bandit.get_active_arms_idx();
			auto active_only_it = active_only.begin();
			for (auto it = active_arm_idx.begin(); it !=active_arm_idx.end(); ++it, ++active_only_it){
				result[*it] = *active_only_it;
			}
			return result;
		}
		template<class T>
		std::vector<T> reorder_to_complete_arm_set(std::vector<T>& some_arms, std::set<unsigned int> used_arms, T fill_value){
			std::vector<T> result(this->bandit.num_arms(), fill_value);
			auto some_arms_it = some_arms.begin();
			for (auto it = used_arms.begin(); it !=used_arms.end(); ++it, ++some_arms_it){
				result[*it] = *some_arms_it;
			}
			return result;
		}
		template<class RNG>
		std::pair<bool, unsigned int> pull_at_least(unsigned int least, RNG& rng){
			bool found = false;
			for(auto it = this->bandit.begin(); it !=this->bandit.end(); ++it){
				if((*it).number_pulled < least){
					found = true;
					break;
				}
			}
			if (!found){
				
				return std::pair<bool, unsigned int>(false,0);
			}
			std::vector<double> values;
			std::vector<unsigned int> idx;
			
			for(auto it = this->bandit.begin(); it !=this->bandit.end(); ++it){
				if((*it).number_pulled < least){
					values.push_back(1.0);
				}
				else{
					values.push_back(0.0);
				}
				idx.push_back((*it).ident);
			}
			unsigned int selected = this->select_max_idx(values, idx, rng);
			return std::pair<bool, unsigned int>(true,selected);
		}
		virtual ~Policy(){}
	};



*/


}} // namespaces
#endif

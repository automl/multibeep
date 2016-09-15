#ifndef MULTIBEEP_BANDIT
#define MULTIBEEP_BANDIT

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <limits>

#include "multibeep/arm/arm.hpp"
#include "multibeep/bandit/arm_info.hpp"
#include "multibeep/util/p_max.hpp"


namespace multibeep{ namespace bandits{


template <typename num_t = double, typename rng_t = std::default_random_engine>
class base{
	protected:
	
		unsigned int num_pulls;
		unsigned int num_active_arms;
		unsigned int num_pulled_arms;
		num_t cummulative_reward;
		std::vector<multibeep::bandits::arm_info<num_t, rng_t> >  arm_infos;
		unsigned int num_dirty_arms;
		bool pmax_dirty;

	public:
	
		base (): num_pulls(0), num_active_arms(0), num_pulled_arms(0), cummulative_reward(0), arm_infos(), num_dirty_arms(0), pmax_dirty(true) {}
	
		virtual ~base() {}
	
	
		/* \brief only method to add arms. There is no way of removing them!
		 * 
		 * The arm is not used directly, but a copy is made to ensure that the arm 
		 * can only be pulled by the bandit class owning it.
		 * */
		unsigned int add_arm(std::shared_ptr<multibeep::arms::base<num_t,rng_t> >arm_ptr){
			unsigned int ident = arm_infos.size();
			// add a copy of the arm
			arm_infos.emplace_back(arm_ptr, ident);
			// reorder arm_infos vector such that all active arms come first
			std::partition(arm_infos.begin(), arm_infos.end(),
						[] (multibeep::bandits::arm_info<num_t, rng_t> a) { return(a.is_active); });
			num_active_arms++;
			num_dirty_arms++;
			return(ident);
		}
		
		/* \brief deactivates an arm by the current index.
		 * 
		 * Inactive arms cannot be pulled, and most policies simply ignore them.
		 * If the arm was not active to begin with nothing changes.
		 * The indices of all arms 'after' index might change as the arms get
		 * partitioned into active/inactive 
		 */
		void deactivate_by_index(unsigned int index){
			if (arm_infos.at(index).is_active){
				// deactivate
				arm_infos.at(index).is_active = false;

				// adjust the number of dirty arms
				if (arm_infos.at(index).dirty)	num_dirty_arms--;
				
				// reorder arm_infos vector such that all active arms come first
				// only touch everything including and beyond the current arm
				std::partition( arm_infos.begin() + index, arm_infos.begin() + num_active_arms,
								[] (multibeep::bandits::arm_info<num_t, rng_t> a) { return(a.is_active); });				
			}
				// adjust number of active arms
				--num_active_arms;
		}
		
		/* \brief deactivates an arm by its unique identifier
		 * 
		 * As the order of the arms might change over time as some of them are
		 * deactivated, this method allows to deactivate an arm using the unique
		 * identifier.
		 */
		void deactivate_by_identifier (unsigned int id){
			for (auto i=0u; i<arm_infos.size(); i++){
				if (arm_infos[i].identifier == id){
					deactivate_by_index(i);
					break;
				}
			}
		}

		/* \brief deactivates all arms whose upper bound is lower than the highest lowest bond.
		 * \param b_ptr a pointer to the bandit
		 * \param delta the confidence value used to compute the lower and upper bounds from the posteriors
		 * \param consider_inactive_arms whether the confidence bounds of already deactivated arms should be used
		 */
		void deactivate_by_confidence_gap (num_t delta, bool consider_inactive_arms){

			// ids = identifiers
			std::vector<unsigned int> ids(num_active_arms, 0);
			// ubs = upper bounds
			std::vector<num_t> ubs (num_active_arms, std::numeric_limits<num_t>::max());
			// llb = largest lower bound
			num_t llb = std::numeric_limits<num_t>::lowest();

			unsigned int i;

			for (i=0; i < num_active_arms; i++){
				const auto &ai = operator[](i);
				ids[i] = ai.identifier;
				num_t mew;
				if (ai.posterior){
					std::tie(mew, ubs[i]) = ai.posterior->support(delta);
					llb = std::max(mew, llb);
				}
			}
			
			if (consider_inactive_arms){
				for (; i < number_of_arms(); i++){
					const auto &ai = operator[](i);
					auto mew = ai.posterior->support(delta);
					llb = std::max(mew.first, llb);
				}
			}
			
			
			
			for (auto i=0u; i < ids.size(); i++){
				if (ubs[i] < llb){
					deactivate_by_identifier(ids[i]);
				}
			}
			std::cout<<std::endl;
		}

		/* \brief deactivates all arms whose upper bound is lower than the highest lowest bond.
		 * \param b_ptr a pointer to the bandit
		 * \param delta the threshold which determines which arm is deactivated
		 */
		void deactivate_by_pmax_threshold (num_t delta){

			// ids = identifiers
			std::vector<unsigned int> ids();
			ids.reserve(num_active_arms);
			
			for (auto i=0u; i < num_active_arms; i++){
				// only deactivate arms when pmax is actually computed
				auto ai = operator[](i);
				if (!std::isnan(ai.pmax))
					if (ai.pmax < delta)
						ids.push_back(ai.identifier);
			}

			for (auto i: ids)
				deactivate_by_identifier(i);
		}

		/* \brief function to automatically remove the n arms with the lowest estimated mean*/
		void deactivate_n_worst( unsigned int n){
			sort_active_arms_by_mean();
			// the following works because we always deactivate the last
			// active arm, i.e., the order of the other does not change!
			while ((n > 0) && (num_active_arms > 0)){
				deactivate_by_index(num_active_arms-1);
				n--;
			}
		}


		/* \brief reactivate an inactive arm.
		 * 
		 * This function reactivates an arm if it was inactive. If it wasn't, nothing happens.
		 * Order of arm indices most likely changes!
		 * */
		void reactivate_by_index(unsigned int index){
			if (! arm_infos.at(index).is_active){
				// reactivate
				arm_infos.at(index).is_active = true;
				// adjust number of active arms
				num_active_arms++;
				
				if (arm_infos.at(index).dirty) num_dirty_arms++;
				
				// reorder arm_infos vector such that all active arms come first
				std::partition(arm_infos.begin(), arm_infos.end(),
					[] (multibeep::bandits::arm_info<num_t, rng_t> a) { return(a.is_active); } );
			}
		}

		/* \brief reactivates an arm by its unique identifier
		 * 
		 * As the order of the arms might change over time as some of them are
		 * deactivated, this method allows to reactivate an arm using the unique
		 * identifier. If the arm was active, nothing happens
		 */
		void reactivate_by_identifier (unsigned int id){
			for (auto i=0u; i<arm_infos.size(); i++){
				if (arm_infos[i].identifier == id){
					reactivate_by_index(i);
					break;
				}
			}
		}

		/* \brief pull selected arm and receive reward.
		 * 
		 * This function handles the pulling of an arm and (usually) the
		 * update of all arm_infos that have to be updated afterwards
		 * */
		num_t pull_by_index (unsigned int index){
			// only active arms can be pulled
			if (!arm_infos.at(index).is_active) return(NAN);
			if (arm_infos[index].num_pulls == 0) num_pulled_arms++;
			num_pulls++;
			num_t r = arm_infos[index].pull();
			cummulative_reward += r;
			num_dirty_arms++;
			pmax_dirty = true;
			return(r);
		}


		/* \brief pull selected arm and receive reward.*/
		num_t pull_by_identifier (unsigned int id){
			for (auto i=0u; i < arm_infos.size(); i++)
				if (arm_infos[i].identifier == id)
					return( pull_by_index(i));
			return(NAN);
		}

		/* \brief makes sure each active arm is pulled a given number of times
		 */
		void min_pull_arms(unsigned int min_num_pulls){
			for (auto i=0u; i < num_active_arms; i++){
				while (operator[](i).num_pulls < min_num_pulls)
					pull_by_index(i);
			}
		}

		void update_active_arm_infos (){
			for (auto i = 0u; i<num_active_arms; i++){
				if (arm_infos[i].dirty)
					update_arm_info(i);
				if (num_dirty_arms == 0)
					break;
			}
		}

		void sort_active_arms_by_mean(){
			
			update_active_arm_infos();
			
			std::sort(arm_infos.begin(), arm_infos.begin()+num_active_arms,
				[] (const multibeep::bandits::arm_info<num_t, rng_t>& a, const multibeep::bandits::arm_info<num_t, rng_t> &b)
				{return(a.estimated_mean > b.estimated_mean);}
				);
		}

		unsigned int number_of_arms() {return(arm_infos.size());}
		unsigned int number_of_active_arms() {return(num_active_arms);}
		
		unsigned int number_of_pulls() {return(num_pulls);}
		unsigned int number_of_pulled_arms() {return(num_pulled_arms);}
		
		
		virtual void update_arm_info(unsigned int index) = 0;
		
		/* \brief only way to access the arm infos to decide which arm to pull next
		 * 
		 * This is a lazy bandit, meaning the arm_infos are only updated
		 * before returning the reference. This should speed things up for
		 * doing batches of pulls from a stochastic policy without updating it
		 */
		const multibeep::bandits::arm_info<num_t, rng_t> &operator[] (unsigned int index){
			if (num_dirty_arms > 0)
				update_arm_info(index);
			// overwrite the pmax value if it is not up-to-date
			if (pmax_dirty)	arm_infos[index].p_max = NAN;
			return(arm_infos[index]);
		}

		/* \brief updates the p_max values of all (active) arms
		 *
		 * This computes the probability of every (active) arm of having
		 * the largest mean based on the posteriors. This is computationally
		 * quite expensive and is not called automatically after every pull or
		 * before an arm_info is requested.
		 *
		 * \param consider_inactive	toggles whether the posteriors of the inactive arms are	also used, and their p_max value is computed
		 * \param delta 			adjusts the integration interval. See multibeep::util::posterior::base::support for more information
		 * \param GL_num_points		number of points used during the Gauss-Legendre Integartion
		 */
		void update_p_max (bool consider_inactive, num_t delta, unsigned int GL_num_points){

			// make sure all arms are up-to-date and simultaniously
			// overwrite the p_max entry with NAN
			for (auto i=0u; i < arm_infos.size(); ++i){
				update_arm_info(i);
				arm_infos[i].p_max = NAN;
			}

			std::vector<std::shared_ptr<multibeep::util::posteriors::base<num_t, rng_t> > > posts;

			// how many arms have to be considered
			unsigned int n = (consider_inactive? arm_infos.size():num_active_arms);
			posts.reserve(n);

			for ( auto i=0u; i < n; ++i){
				posts.emplace_back(arm_infos[i].posterior);
			}

			auto pmax_vector = multibeep::util::pmax::compute_pmax_all<num_t, rng_t> (posts, delta, GL_num_points);

			for (auto i=0u; i < n; ++i){
				arm_infos[i].p_max = pmax_vector[i];
			}

			pmax_dirty = false;
		}

};


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


}} // namespaces
#endif

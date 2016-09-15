#ifndef MULTIBEEP_UTIL_FRIEDMAN
#define MULTIBEEP_UTIL_FRIEDMAN


#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/chi_squared.hpp>



namespace multibeep{ namespace util{ namespace friedman{


template <typename vv_t>
void print_vv(vv_t vv){
	for (auto &v: vv){
		for ( auto e: v)
			std::cout<<e<<" ";
		std::cout<<std::endl;
	}
}


template <typename vv_t>
void print_vv_t(vv_t vv){
	for (auto j=0u; j<vv[0].size(); j++){
		for (auto i=0u; i<vv.size(); i++)
			std::cout<<vv[i][j]<<" ";
		std::cout<<std::endl;
	}
}



template <typename num_t>
/*\brief non-parametric friedman test to identify underperforming arms
 * 
 * \param performances vector of pointers to reward vectors (to avoid copies)
 * \return vector of vectors with the mean ranks
 */
std::vector<std::vector<num_t> > compute_ranks( std::vector< std::vector<num_t>* >performances){
	/* notation follows 
	 * Birattari, Mauro, et al. "F-Race and iterated F-Race: An overview."
	 * in Experimental methods for the analysis of optimization algorithms. Springer Berlin Heidelberg, 2010. 311-336.
	 * http://iridia.ulb.ac.be/~mbiro/paperi/BirYuaBalStu2010emaoa.pdf
	 */
	 
	 // number of arms
	unsigned int m = performances.size();
	// number of pulls
	unsigned int k = std::numeric_limits<unsigned int>::max();
	
	
	for (auto &p : performances)
		k = std::min<unsigned int>(k, p->size());
	
	// vector holding all the ranks
	std::vector<std::vector<num_t> > ranks(m, std::vector<num_t> (k,0) );
	
	std::vector<unsigned int> indices (m);
	std::iota(indices.begin(), indices.end(),0);
	
	//compute ranks for each 'round'
	for (auto i=0u; i<k; i++){
		std::sort(
			indices.begin(), indices.end(),
			[performances, i] (unsigned int a, unsigned int b){return((*(performances[a]))[i] > (*(performances[b]))[i]);}
		);

		for (auto it1=indices.begin(); it1!=indices.end(); ){
			auto it2 = it1; it2++;
			num_t n = 1, sum = std::distance(indices.begin(), it1)+1;
			while ( (it2 != indices.end()) && (*(performances[*it2]))[i] == (*(performances[*it1]))[i]){
				n+=1;
				sum += std::distance(indices.begin(), it2)+1;
				it2++;
			}
			for (auto it3=it1; it3!=it2; it3++)
				ranks[*it3][i] = sum/n;
			it1 = it2;
		}
	}
	return(ranks);
}


template <typename num_t>
/*\brief non-parametric friedman test to identify underperforming arms
 * 
 * \param performances vector of pointers to reward vectors (to avoid copies)
 * \param alpha confidence level, usually 0.05
 * \return (possibly empty) vector of indices for arms that perform significantly worse then the 'best' arm
 */
std::vector<unsigned int> friedman_test( std::vector< std::vector<num_t>* >performances, num_t alpha){
	/* notation follows 
	 * Birattari, Mauro, et al. "F-Race and iterated F-Race: An overview."
	 * in Experimental methods for the analysis of optimization algorithms. Springer Berlin Heidelberg, 2010. 311-336.
	 * http://iridia.ulb.ac.be/~mbiro/paperi/BirYuaBalStu2010emaoa.pdf
	 */
	// the return vector
	std::vector<unsigned int> rv();

	auto ranks = compute_ranks(performances);
	 // number of arms
	num_t m = ranks.size();
	// number of pulls
	num_t k = ranks[0].size();

	std::vector<unsigned int> sum_R(m,0), sum_R2(m,0);
	
	// compute simple statistics
	for (auto i=0u; i<m; ++i){
		for (auto j=0u; j<k; ++j){
			sum_R [i] += ranks[i][j];
			sum_R2[i] += ranks[i][j]*ranks[i][j];
		}
	}
	
	// compute T as the quantity determening whether the ranking is consistent with the null-hypothesis (all arms perform equally)
	num_t T_numerator(0), T_denominator(0);
	
	for (auto i=0u; i<m; ++i){
		T_numerator   += std::pow(sum_R[i] - k*(m+1)/2, 2);
		T_denominator += sum_R2[i];
	}
	T_numerator   *= (m-1);
	T_denominator -= k*m*(m+1)*(m+1)/4;
	
	auto T = T_numerator/T_denominator;
	


	// T follows approximately a χ² distribution
	auto chi_s = boost::math::chi_squared_distribution<num_t>(m-1);
	// the null-hypothesis should be rejected if this value is larger that the (1-alpha)-quantile
	if (T > boost::math::quantile(chi_s, 1-alpha)){
		// in that case, do a post hoc analysis (following F-Race: a la Conover) by pairwise comparison with the lowest mean rank arm
		
		// note, the paper doesn't specify the degrees of freedom!
		auto students_t = boost::math::students_t_distribution<num_t>(m-1);
		auto threshold = quantile(students_t, 1-alpha/2);
		
		
		// find arm with lowest average rank
		auto min_mean_rank = *(std::min_element(sum_R.begin(), sum_R.end()));

		auto Z = 0;
		for (auto &r2: sum_R2)	Z +=r2;
		Z -= k*m*(m+1)*(m+1)/4;
		Z *= 2*k*(1-T/(k*(m-1)));
		Z /= (k-1)*(m-1);
		Z = std::sqrt(1./Z);
		
		
		//compare every other arm against the 'best'
		for (auto i=0u; sum_R.size(); i++){
			// add the ones that perform significantly worse to the return vector
			if (Z*std::abs(sum_R[i] - min_mean_rank) > threshold ) rv.push_back(i);
		}
	}
	return(rv);
}




}}}
#endif

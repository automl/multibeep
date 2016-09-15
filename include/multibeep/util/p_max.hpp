#ifndef MULTIBEEP_UTIL_PMAX
#define MULTIBEEP_UTIL_PMAX

#include <vector>
#include <random>
#include <algorithm>
#include <iterator>

#include <multibeep/util/posteriors.hpp>
#include <gauss_legendre/gauss_legendre.c>


namespace multibeep{ namespace util{ namespace pmax{


template<typename num_t = double, typename rng_t = std::default_random_engine>
using it_t = typename std::vector<std::shared_ptr<multibeep::util::posteriors::base<num_t, rng_t> > >::iterator;

template<typename num_t = double, typename rng_t = std::default_random_engine>
using post_vector_t = typename std::vector<std::shared_ptr<multibeep::util::posteriors::base<num_t, rng_t> > >;



template<typename num_t = double, typename rng_t = std::default_random_engine>
num_t wrapper_function( num_t x, void* data){
	std::pair<it_t<num_t, rng_t>, it_t<num_t, rng_t> > * it_pair = static_cast<std::pair<it_t<num_t, rng_t>, it_t<num_t, rng_t> >*>(data);

	auto it=it_pair->first;
	num_t res = (*it)->pdf(x);

	for( ++it ; it != it_pair->second; ++it)
		res *= (*it)->cdf(x);
	return(res);
}



template<typename num_t = double, typename rng_t = std::default_random_engine>
num_t compute_pmax_for (unsigned int index, post_vector_t<num_t, rng_t>posts, num_t delta, unsigned int number_of_points){

	num_t num_arms = posts.size();

	// unknown arms get a default p_max of num_invalids/total_num_arms
	if (!(posts[index])){
		return( ((num_t) 1.) /  num_arms );
	}

	// swap the inderesting index with the first
	std::swap(posts[0], posts[index]);

	// remove all invalid posteriors
	auto new_end = std::partition(++posts.begin(), posts.end(),
		[] (std::shared_ptr<multibeep::util::posteriors::base<num_t, rng_t> > ptr) {return(ptr);});

	num_t num_invalid = std::distance(new_end, posts.end());

	auto it_pair = std::pair<it_t<num_t, rng_t>, it_t<num_t, rng_t> >(posts.begin(), new_end);

	// get integration interval
	auto support = posts[0]->support(delta);

	// integrate using Gauss Legendre integration (part of GSL actually)
	num_t approx = gauss_legendre(number_of_points,wrapper_function<num_t, rng_t>,&it_pair ,support.first,support.second);

	// adjust for unknown arms
	return(approx * (1. - ((num_t) num_invalid) /  ((num_t) posts.size())));
}


template<typename num_t>
void normalize(std::vector<num_t> & vec){
	num_t sum = 0;
	for (auto &v: vec) sum += v;
	for (auto &v: vec)  v  /= sum;
}




template<typename num_t = double, typename rng_t = std::default_random_engine>
std::vector<num_t> compute_pmax_all (post_vector_t<num_t, rng_t> &posts, num_t delta, unsigned int number_of_points){

	std::vector<num_t> pmax_values(posts.size(), 0.);

	for (auto i=0u; i<posts.size(); ++i)
		pmax_values[i] = compute_pmax_for<num_t, rng_t>(i, posts, delta, number_of_points);

	normalize(pmax_values);

	return(pmax_values);
}



}}}
#endif

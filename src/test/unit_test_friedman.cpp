#include <vector>
#include <iostream>


#include <boost/test/unit_test.hpp>
#include "multibeep/util/friedman_test.hpp"


typedef double num_t;


BOOST_AUTO_TEST_CASE(test_ranking){

	std::vector<num_t> p1 = { 0, 0, 0, 0, 0, 0, 0, 0, 0};
	std::vector<num_t> p2 = { 0, 1,-1, 1, 0, 1,-1, 0, 1};
	std::vector<num_t> p3 = {-1, 1,-1, 1, 1, 1,-1, 0, 2};	
	std::vector<num_t> p4 = { 1, 0, 1, 0, 1, 1,-1, 0, 3};
	
	std::vector< std::vector<num_t>* >performances = {&p1, &p2, &p3, &p4};
	
	
	
	
	auto ranks = multibeep::util::friedman::compute_ranks(performances);
	
	std::vector< std::vector<num_t> > true_ranks = {
			{2.5, 3.5, 2  , 3.5, 3.5, 4  , 1  , 2.5, 4  },
			{2.5, 1.5, 3.5, 1.5, 3.5, 2  , 3  , 2.5, 3  },
			{4  , 1.5, 3.5, 1.5, 1.5, 2  , 3  , 2.5, 2  },
			{1  , 3.5, 1  , 3.5, 1.5, 2  , 3  , 2.5, 1  }
		};
		
	BOOST_CHECK_EQUAL_COLLECTIONS(ranks[0].begin(), ranks[0].end(), 
						true_ranks[0].begin(), true_ranks[0].end());

	BOOST_CHECK_EQUAL_COLLECTIONS(ranks[1].begin(), ranks[1].end(), 
						true_ranks[1].begin(), true_ranks[1].end());

	BOOST_CHECK_EQUAL_COLLECTIONS(ranks[2].begin(), ranks[2].end(), 
						true_ranks[2].begin(), true_ranks[2].end());

	BOOST_CHECK_EQUAL_COLLECTIONS(ranks[3].begin(), ranks[3].end(), 
						true_ranks[3].begin(), true_ranks[3].end());
}


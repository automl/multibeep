cdef extern from "<random>" namespace "std":
	cdef cppclass default_random_engine:
		default_random_engine(int)
		void seed(int)
		double operator() ()


ctypedef double float_t
ctypedef default_random_engine rand_t

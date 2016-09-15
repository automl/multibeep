#ifndef MULTIBEEP_UTIL_STATISTICS
#define MULTIBEEP_UTIL_STATISTICS


namespace multibeep{ namespace util{ namespace statistics{


template<typename num_type>
class running_statistics{
  private:
	long unsigned int N;
	num_type m, v;
  public:
	running_statistics(): N(0), m(0), v(0) {}
	
	void operator() (num_type x){
		++N;
		num_type delta = x - m;
		// adjust mean
		m += delta/N;
		// adjust variance
		v += delta*(x-m);
	}
	
	long unsigned int number_of_points() const {return(N);}
	num_type mean() const { return( (N>0)?m:NAN);}
	num_type variance() const {return((N>1)?std::max<double>(0.,v/(N-1)) : NAN);}
};



template <typename num_type>
class running_covariance{
  private:
	long unsigned int N;
	num_type m1, m2;
	num_type cov;

  public:
	running_covariance(): N(0), m1(0), m2(0), cov(0) {}
	
	void operator() (num_type x1, num_type x2){
		N++;
		num_type delta1 = (x1-m1)/N;
		m1 += delta1;
		num_type delta2 = (x2-m2)/N;
		m2 += delta2;
		
		cov += (N-1) * delta1 * delta2 - cov/N;
	}
	
	long unsigned int number_of_points(){return(N);}
	num_type covariance(){return(num_type(N)/num_type(N-1)*cov);}
};



}}}
#endif

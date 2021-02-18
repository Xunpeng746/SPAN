#ifndef MEMFACTORY_HPP
#define MEMFACTORY_HPP
#include<vector>
#include<mkl.h>
#include <ctime>
class MemFactory{
    public:
        std::vector<double*> double_ptrs;
        std::vector<int*> int_ptrs;
        MemFactory(){}
        ~MemFactory(){
            for(int i = 0; i < int_ptrs.size(); ++i){
                int* ptr = int_ptrs[i];
                if(ptr != nullptr){
                    mkl_free(ptr);
                }
            }
            for(int i = 0; i < double_ptrs.size(); ++i){
                double* ptr = double_ptrs[i];
                if(ptr != nullptr){
                    mkl_free(ptr);
                }
            }
        }
        inline int* malloc_int(size_t n)
        {
            int* ptr = (int*)mkl_malloc(sizeof(int)*n, 64);
            int_ptrs.push_back(ptr);
            return ptr;
        }
        inline double* malloc_double(size_t n){
            double* ptr = (double*)mkl_malloc(sizeof(double)*n, 64);
            double_ptrs.push_back(ptr);
            return ptr;
        }
		inline double* get_randn(int num, double mu, double sigma) {
			double* data = (double*)mkl_malloc(sizeof(double)*num, 64);
			srand((int)time(0));
			VSLStreamStatePtr stream_MCG31;
			vslNewStream(&stream_MCG31, VSL_BRNG_MCG31, rand()%1234567);
			vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream_MCG31, num, data, mu, sigma);
			vslDeleteStream(&stream_MCG31);
			double_ptrs.push_back(data);
			return data;
		}
		inline int* get_rand_choice(int num, int l, int r) {
			int* idx = (int*)mkl_malloc(sizeof(int)*(num), 64);
			srand((int)time(0));
			VSLStreamStatePtr stream_uniform;
			vslNewStream(&stream_uniform, VSL_BRNG_MT19937, rand()%1234567);
			//randint(num, idx, l, r, stream_uniform);
			viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream_uniform, num, idx, l, r);
			vslDeleteStream(&stream_uniform);
			int_ptrs.push_back(idx);
			return idx;
		}
};
#endif

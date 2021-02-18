#ifndef CONFIG_H
#define CONFIG_H
#include <mkl.h>
#include <cstdio>
#include<utility>
#include<cmath>
#include<cstdlib>
#include<string>
#include <omp.h>
#include<chrono>
#include "json.hpp"
//#include<algorithm>
//#include "Param.hpp"
//#include "Records.hpp"

#define asum cblas_dasum
#define dot cblas_ddot
#define gemm cblas_dgemm
#define axpby cblas_daxpby
#define axpy cblas_daxpy
#define gemv cblas_dgemv
#define ger cblas_dger
#define scal cblas_dscal
#define nrm2(n,a) cblas_ddot(n, a, 1, a, 1)
#define squared_norm(n, x) cblas_ddot(n, x, 1, x, 1)
#define mem_copy cblas_dcopy
//#define mean(n, x) (cblas_dasum(n, x, 1)/n)
#define mi(x, y) (((x)<(y))?(x):(y))
#define ma(x, y) (((x)>(y))?(x):(y))
#define Trans CblasTrans
#define NoTrans CblasNoTrans
#define RowMajor CblasRowMajor
const size_t incx = 1;
const size_t incy = 1;
const double zero = 0.0;
#define mem_zero(n, a) cblas_dscal(n, zero, a, 1)
#define RAND_SEED 1
#define transpose mkl_dimatcopy
#define RAND_MU 0.0
#define RAND_SIGMA 1.0
#define randint(n, data, l, r, stream) viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, data, l, r);
#define randn(n, data, mu, sigma, stream) vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, n, data, mu, sigma);
#define LRCT_loss 0.6650962888284543
#define LRCOVTYPE_6_LOSS 0.666991689674
#define LRCOVTYPE_7_LOSS 0.6650395375576065
#define LRCOVTYPE_5_LOSS 0.6705471636550693 
#define WEBSPAM_4_LOSS 0.2938208268476120
#define WEBSPAM_5_LOSS 0.2261548071555874
#define WEBSPAM_6_LOSS 0.2004491605134106
#define YELP_6_LOSS 0.2026297341510391
#define LR49_4_loss 0.1639146291170644
#define LR49_3_loss 0.321516197743060
#define LRSynthetic_loss 0.6116597184627440
#define min_time 0.05
extern bool PRINT_HESS;
#include "easylogging++.h"
//
using json = nlohmann::json;
extern double eps;

extern size_t THREAD_NUM_MKL;
extern double min_loss;
extern double delta_lambda;
extern size_t THREAD_NUM_OMP;
extern json params_json;
using namespace std;
inline double sub(const  chrono::time_point<chrono::high_resolution_clock>& ed, const chrono::time_point<chrono::high_resolution_clock>& st) {
	return (double)(chrono::duration_cast<chrono::duration<double>>(ed - st)).count();
}
inline double expit(int n, double* x, double* out) {
	vdExp(n, x, out);
	vdLinearFrac(n, out, out, 1.0, 0.0, 1.0, 1.0, out);
	//return exp(x) / (1.0 + exp(x));
}
inline double expit(double x) {
	return 1.0 / (1.0 + exp(-x));
	//return exp(x) / (1.0 + exp(x));
}

inline double log_logistic(double x) {
	if (x > 0) {
		return -log(1 + exp(-x));
	}
	else {
		return x - log(1 + exp(x));
	}
}


#endif
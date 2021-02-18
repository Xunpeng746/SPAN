#include "config.h"
#include "Model.h"
#include "Logistic.h"
#include "LSvm.h"
#include "Lant.h"
#include "Util.h"
#include<cstdlib>
// extern size_t MAX_DIM;
namespace Test {
	void test_norm();

	void test_get_hessian_mm();
	void test_get_hess_vt_mm();
	void test_get_gradient_mm();
	void test_logistic_get_gradient(double* X, double* Y, int N);
	void test_logistic_hess_vt(double* X, double* Y, int N);
	void test_mkl_gemm(int N = 10);
	void test_get_randn(int N = 10);
	void test_get_rand_choice(int N = 10);
	void test_init_randn(int N = 10);
	void test_mkl_lq(int M = 3, int N = 5);
	void test_mkl_svd(int N = 10);
	void test_logistic_hessian(double* X, double* Y, int N, int sample_size);
	void test_lant_hess_dot_approx(double* X, double* Y, int N, int sample_size, int rank = 10);
	void test_logistic_get_gradient_multi_ndim(double* X, double* Y, int N, int rank);
	void test_run_stage_a_4_3(double* X, double* Y, int N, int rank);
	void test_run_stage_a_5_3(double* X, double* Y, int N, int rank);
	void test_ksvd();
	void test_random_ksvd();
	void test_symm_inv(int N);
	void test_inv(int N);
	void test_openmpi_mkl(int m, int n, int thread_num = 5);
	void test_openmpi(int m, int n, int thread_num = 5);
	void test_model();
}
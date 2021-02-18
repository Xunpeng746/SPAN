#include "Logistic.h"


void Logistic::get_hess_vt(int idx, const double * vt, double * hess_vt, const double * weights, int vndim, bool add) 
{
	if (weights == nullptr) weights = m_weights;
	double yxw = dot(MAX_DIM, weights, 1, X + MAX_DIM * idx, 1)*Y[idx];
	double sigmoid = expit(yxw);
	double hess_value = sigmoid * (1 - sigmoid);
	double init_value = (double)add;
	if (vndim == 1) {
		double value = hess_value * dot(MAX_DIM, vt, 1, X + MAX_DIM * idx, 1);
		axpby(MAX_DIM, m_lambda[0], vt, 1, init_value, hess_vt, 1);
		axpby(MAX_DIM, value, X + MAX_DIM * idx, 1, 1.0, hess_vt, 1);
	}else {	
		throw "vndim>1";
	}
}
void Logistic::get_hess_vt_mm(double * data_x, double * data_y, int n_samples, const double * vt, 
double * hess_vt, const double * weights, int vndim, int _power_iter, double* B)
{
	if (weights == nullptr) weights = m_weights;
	double* sigmoid = (double*)mkl_malloc(sizeof(double)*n_samples, 64);
	double* hess_value = (double*)mkl_malloc(sizeof(double)*n_samples, 64);
	double* dx_vt = (double*)mkl_malloc(sizeof(double)*n_samples*vndim, 64);

	// sigmoid == (1.0/(1+exp(wxy)))
	gemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, n_samples, MAX_DIM, 1.0, weights, MAX_DIM,
		data_x, MAX_DIM, 0, sigmoid, n_samples);
	

	vdMul(n_samples, sigmoid, data_y, sigmoid);
	vdExp(n_samples, sigmoid, sigmoid);
	vdLinearFrac(n_samples, sigmoid, sigmoid, 0, 1.0, 1.0, 1.0, sigmoid);

	// hess_value = sigmoid*(1-sigmoid)
	mem_copy(n_samples, sigmoid, 1, hess_value, 1);
	vdMul(n_samples, sigmoid, sigmoid, sigmoid);
	vdSub(n_samples, hess_value, sigmoid, hess_value);
	vdSqrt(n_samples, hess_value, hess_value);


	for (int i = 0; i < n_samples; ++i) {
		scal(MAX_DIM, hess_value[i], data_x + i * MAX_DIM, 1);
	}
	// dx_vt=(x@vt)*hess_value
	gemm(CblasRowMajor, CblasNoTrans, CblasTrans, vndim, n_samples, MAX_DIM, 1.0, vt, MAX_DIM, data_x, MAX_DIM, 0, dx_vt, n_samples);
	// hess_vt = lambda*vt + 1.0/n_smples * x @dx_vt
	axpby(MAX_DIM*vndim, m_lambda[0], vt, 1, 0, hess_vt, 1);
	gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, vndim, MAX_DIM, n_samples, 1.0/n_samples, dx_vt, n_samples, data_x, MAX_DIM, 1.0, hess_vt, MAX_DIM);
	
	if(~_power_iter){
		//just for lant optimization

		double* tau = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
		//2q power
		for(int i = 0; i < _power_iter; ++i){
			// mem_copy(MAX_DIM*vndim, hess_vt, 1, dx_vt, 1);
			gemm(CblasRowMajor, CblasNoTrans, CblasTrans, vndim, n_samples, MAX_DIM, 1.0, hess_vt, MAX_DIM, data_x, MAX_DIM, 0, dx_vt, n_samples);
			gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, vndim, MAX_DIM, n_samples, 1.0/n_samples, dx_vt, n_samples, data_x, MAX_DIM, m_lambda[0], hess_vt, MAX_DIM);
			gemm(CblasRowMajor, CblasNoTrans, CblasTrans, vndim, n_samples, MAX_DIM, 1.0, hess_vt, MAX_DIM, data_x, MAX_DIM, 0, dx_vt, n_samples);
			gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, vndim, MAX_DIM, n_samples, 1.0/n_samples, dx_vt, n_samples, data_x, MAX_DIM, m_lambda[0], hess_vt, MAX_DIM);
			// LAPACKE_dgelqf(CblasRowMajor, vndim, MAX_DIM, hess_vt, MAX_DIM, tau);
			// LAPACKE_dorglq(CblasRowMajor, vndim, MAX_DIM, vndim, hess_vt, MAX_DIM, tau);
		}

		//orthogonalization
		LAPACKE_dgelqf(CblasRowMajor, vndim, MAX_DIM, hess_vt, MAX_DIM, tau);
		LAPACKE_dorglq(CblasRowMajor, vndim, MAX_DIM, vndim, hess_vt, MAX_DIM, tau);
	
		// to get B
		// dx_vt = q_matrix@X.T
		// B = q_matrix@(X.T@X/n_samples + lambda@I)@q_matrix.T
		//   = dx_vt@dx_vt.T/n_samples + lambda*q_matrix@q_matrix.T
		gemm(CblasRowMajor, CblasNoTrans, CblasTrans, vndim, n_samples, MAX_DIM, 1.0, hess_vt, MAX_DIM, data_x, MAX_DIM, 0, dx_vt, n_samples);
		gemm(CblasRowMajor, CblasNoTrans, CblasTrans, vndim, vndim, MAX_DIM, m_lambda[0], hess_vt, MAX_DIM, hess_vt, MAX_DIM, 0, B, vndim);
		gemm(CblasRowMajor, CblasNoTrans, CblasTrans, vndim, vndim, n_samples, 1.0/n_samples, dx_vt, n_samples, dx_vt, n_samples, 1.0, B, vndim);
		mkl_free(tau);
	}
	
	mkl_free(dx_vt);
	mkl_free(sigmoid);
	mkl_free(hess_value);
}
void Logistic::get_hessian(const int idx, double * hess, const double * weights, bool add)
{
	if (weights == nullptr) weights = m_weights;
	double yxw = dot(MAX_DIM, weights, 1, X + MAX_DIM * idx, 1)*Y[idx];
	double sigmoid = expit(yxw);
	double hess_value = sigmoid * (1 - sigmoid);
	double init_value = (double)add;

	if (!add) { mem_zero(MAX_DIM*MAX_DIM, hess); }
	ger(CblasRowMajor, MAX_DIM, MAX_DIM, hess_value, X + MAX_DIM * idx, 1, X + MAX_DIM * idx, 1, hess, MAX_DIM);
	if (!add) {
		for (int i = 0; i < MAX_DIM; ++i) hess[i*MAX_DIM + i] += m_lambda[0];
	}
}



void Logistic::get_hessian_mm(double * data_x, double * data_y, int n_samples, double * hess, const double * weights)
{
	if (weights == nullptr) weights = m_weights;
	double* sigmoid = (double*)mkl_malloc(sizeof(double)*n_samples, 64);
	double* hess_value = (double*)mkl_malloc(sizeof(double)*n_samples, 64);
	double* dx = (double*)mkl_malloc(sizeof(double)*n_samples*MAX_DIM, 64);
	mem_copy(n_samples*MAX_DIM, data_x, 1, dx, 1);

	// sigmoid = expit(w@X.T*Y)
	gemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, n_samples, MAX_DIM, 1.0, weights, MAX_DIM,
		data_x, MAX_DIM, 0, sigmoid, n_samples);
	
	
	vdMul(n_samples, sigmoid, data_y, sigmoid);
	vdExp(n_samples, sigmoid, sigmoid);
	vdLinearFrac(n_samples, sigmoid, sigmoid, 0, 1.0, 1.0, 1.0, sigmoid);

	// hess_value = sigmoid*(1-sigmoid)
	// dx = X*sqrt(hess_value)
	mem_copy(n_samples, sigmoid, 1, hess_value, 1);
	vdMul(n_samples, sigmoid, sigmoid, sigmoid);
	vdSub(n_samples, hess_value, sigmoid, hess_value);
	// vdSqr(n_samples, hess_value, hess_value);
	//mkl_set_num_threads_local(1);
	//#pragma	omp parallel for num_threads(THREAD_NUM_OMP)
	for (int i = 0; i < n_samples; ++i) {
		scal(MAX_DIM, hess_value[i], dx + i * MAX_DIM, 1);
	}

	//mkl_set_num_threads_local(0);

	//init hessian, hessian = lambda*I
	mem_zero(MAX_DIM*MAX_DIM, hess);
	for (int i = 0; i < MAX_DIM; ++i) {
		hess[i*MAX_DIM + i] = m_lambda[0];
	}
	//hess += 1.0/n_samples* (dx.T@dx)
	gemm(CblasRowMajor, CblasTrans, CblasNoTrans, MAX_DIM, MAX_DIM, n_samples, 1.0 / n_samples, dx, MAX_DIM, data_x, MAX_DIM, 1.0, hess, MAX_DIM);

	mkl_free(dx);
	mkl_free(sigmoid);
	mkl_free(hess_value);
}

double Logistic::get_loss(const double* weights) const
{
	if (weights == nullptr) weights = m_weights;
	double loss = 0;
	for (int i = 0; i < N; ++i) {
		double yxw = dot(MAX_DIM, weights, 1, X + MAX_DIM * i, 1)*Y[i];
		loss -= log_logistic(yxw);
	}
	loss /= N;
	loss += 0.5*m_lambda[0]*squared_norm(MAX_DIM, weights);
	return loss;
}

void Logistic::get_gradient(int idx, double * grad, const double* weights, int wndim, bool add) 
{
	if (weights == nullptr) weights = m_weights;
	double init_value = (double)add;
	if (wndim == 1) {
		double yxw = Y[idx] * dot(MAX_DIM, weights, 1, X + MAX_DIM * idx, 1);
		double sigmoid = expit(yxw);
		sigmoid = (sigmoid - 1)*Y[idx];
		axpby(MAX_DIM, m_lambda[0], weights, 1, init_value, grad, 1);
		axpby(MAX_DIM, sigmoid, X + MAX_DIM * idx, 1, 1.0, grad, 1);
		
	}else {
		throw "wndim>1";
	}
}

void Logistic::get_gradient_mm(double* data_x, double* data_y, int n_samples, double * grad, const double * weights, int wndim)
{

	if (weights == nullptr) weights = m_weights;
	int len = wndim * n_samples;
	double* sigmoid = (double*)mkl_malloc(sizeof(double)*len, 64);
	gemm(CblasRowMajor, CblasNoTrans, CblasTrans, wndim, n_samples, MAX_DIM, 1.0, weights, MAX_DIM,
		data_x, MAX_DIM, 0, sigmoid, n_samples);
	mkl_set_num_threads_local(1);
	#pragma	omp parallel for num_threads(THREAD_NUM_OMP)
	for (int i = 0; i < wndim; ++i) {
		vdMul(n_samples, sigmoid + i * n_samples, data_y, sigmoid + i * n_samples);
	}
	mkl_set_num_threads_local(0);
	vdExp(len, sigmoid, sigmoid);
	vdLinearFrac(len, sigmoid, sigmoid, 0,-1.0, 1.0, 1.0, sigmoid);
	mkl_set_num_threads_local(1);
	#pragma	omp parallel for num_threads(THREAD_NUM_OMP)
	for (int i = 0; i < wndim; ++i) {
		vdMul(n_samples, sigmoid + i * n_samples, data_y, sigmoid + i * n_samples);
	}
	mkl_set_num_threads_local(0);
	mem_copy(wndim*MAX_DIM, weights, 1, grad, 1);
	gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, wndim, MAX_DIM, n_samples, 1.0 / n_samples, sigmoid,
		n_samples, data_x, MAX_DIM, m_lambda[0], grad, MAX_DIM);
	mkl_free(sigmoid);
}
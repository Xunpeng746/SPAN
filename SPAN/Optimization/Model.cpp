#include "Model.h"



void Model::get_full_gradient(double* grad, const double* weights, int wndim){
	if (use_mm) {
		if (wndim > 1) {
			throw "wndim > 1 when get full gradient";
		}
		get_gradient_mm(this->X, this->Y, this->N, grad, weights, wndim);
	}
	// else {
		
	// 	mem_zero(wndim*MAX_DIM, grad);
	// 	for (int i = 0; i < N; ++i) {
	// 		get_gradient(i, grad, weights, wndim, true);
	// 	}
	// 	scal(MAX_DIM*wndim, 1.0 / N, grad, 1);
	// }
}
void Model::get_gradient(const int idx[], int n_samples, double * grad, const double * weights, int wndim) 
{
	if (weights == nullptr) weights = m_weights;
	if (use_mm) {
		double *data_x = (double*)mkl_malloc(sizeof(double)*MAX_DIM*n_samples, 64);
		double *data_y = (double*)mkl_malloc(sizeof(double)*n_samples, 64);
		
		// mkl_set_num_threads_local(1);
		// #pragma	omp parallel for num_threads(THREAD_NUM_OMP)
		for (int i = 0; i < n_samples; ++i) {
			mem_copy(MAX_DIM, X + idx[i] * MAX_DIM, 1, data_x + i * MAX_DIM, 1);
			data_y[i] = Y[idx[i]];
		}
		// mkl_set_num_threads_local(0);
		get_gradient_mm(data_x, data_y, n_samples, grad, weights, wndim);
		
		mkl_free(data_x);
		mkl_free(data_y);
	}
	// else {
	// 	mem_zero(MAX_DIM*wndim, grad);
	// 	//double* div_grad = (double*)mkl_malloc(sizeof(double)*wndim*MAX_DIM, 64);
	// 	for (int i = 0; i < n_samples; ++i) {
	// 		for (int j = 0; j < wndim; ++j) {
	// 			get_gradient(idx[i], grad + MAX_DIM * j, weights + MAX_DIM * j, 1, true);
	// 		}

	// 	}
	// 	scal(MAX_DIM*wndim, 1.0 / n_samples, grad, 1);
	// }

}


void Model::get_hess_vt(const int idx[], int n_samples, const double * vt, double * hess_vt,
	const double * weights, int vndim, int _power_iter, double* B) 
{
	if (weights == nullptr) weights = m_weights;
	if (use_mm) {
		double *data_x = (double*)mkl_malloc(sizeof(double)*MAX_DIM*n_samples, 64);
		double *data_y = (double*)mkl_malloc(sizeof(double)*MAX_DIM*n_samples, 64);
		mkl_set_num_threads_local(1);
		#pragma	omp parallel for num_threads(THREAD_NUM_OMP)
		for (int i = 0; i < n_samples; ++i) {
			mem_copy(MAX_DIM, X + idx[i] * MAX_DIM, 1, data_x + i * MAX_DIM, 1);
			data_y[i] = Y[idx[i]];
		}
		mkl_set_num_threads_local(0);

		get_hess_vt_mm(data_x, data_y, n_samples, vt, hess_vt, weights, vndim, _power_iter, B);
		mkl_free(data_x);
		mkl_free(data_y);
	}
	// else {
	// 	mem_zero(vndim*MAX_DIM, hess_vt);
	// 	for (int i = 0; i < n_samples; ++i) {
	// 		for (int j = 0; j < vndim; ++j) {
	// 			get_hess_vt(idx[i], vt + MAX_DIM * j, hess_vt + MAX_DIM * j, weights, 1, true);
	// 		}

	// 	}
	// 	scal(vndim*MAX_DIM, 1.0 / n_samples, hess_vt, 1);
	// }
}

 void Model::get_full_hessian(double* hess, const double* weights){
 	if (weights == nullptr) weights = m_weights;
 	if (use_mm) {
 		get_hessian_mm(X, Y, N, hess, weights);
 	}
	//  else {
 	// 	mem_zero(MAX_DIM*MAX_DIM, hess);
 	// 	for (int i = 0; i < N; ++i) {
 	// 		get_hessian(i, hess, weights, true);
 	// 	}
 	// 	scal(MAX_DIM*MAX_DIM, 1.0 / N, hess, 1);
 	// 	for (int i = 0; i < MAX_DIM; ++i) hess[MAX_DIM*i + i] += m_lambda[0];
 	// }	
 }
void Model::get_hessian(const int idx[], int n_samples, double * hess, const double * weights)
{
	if (weights == nullptr) weights = m_weights;
	if (use_mm) {
		double *data_x = (double*)mkl_malloc(sizeof(double)*MAX_DIM*n_samples, 64);
		double *data_y = (double*)mkl_malloc(sizeof(double)*MAX_DIM*n_samples, 64);
		// mkl_set_num_threads_local(1);
		// #pragma	omp parallel for num_threads(THREAD_NUM_OMP)
		for (int i = 0; i < n_samples; ++i) {
			mem_copy(MAX_DIM, X + idx[i] * MAX_DIM, 1, data_x + i * MAX_DIM, 1);
			data_y[i] = Y[idx[i]];
		}
		// mkl_set_num_threads_local(0);
		get_hessian_mm(data_x, data_y, n_samples,hess, weights);
		mkl_free(data_x);
		mkl_free(data_y);
	}
	// else {
	// 	mem_zero(MAX_DIM*MAX_DIM, hess);
	// 	for (int i = 0; i < n_samples; ++i) {
	// 		get_hessian(idx[i], hess, weights, true);
	// 	}
	// 	scal(MAX_DIM*MAX_DIM, 1.0 / n_samples, hess, 1);
	// 	for (int i = 0; i < MAX_DIM; ++i) hess[MAX_DIM*i + i] += m_lambda[0];
	// }
}



void Model::set_init_weight(const double * weights)
{
	update_model(weights);
}

double * Model::get_model() const
{
	return m_weights;
}

void Model::init_model(double* weights)
{
	if (weights == nullptr){
		Util::get_randn(m_weights, MAX_DIM, 0, 1);
	}else {
		update_model(weights);
	}
}

double* Model::get_params() const
{
	return m_lambda;
}

void Model::step(double step_size, double * data)
{
	axpy(MAX_DIM, -step_size, data, 1, m_weights, 1);
}

void Model::update_model(const double * new_weights)
{
	mem_copy(MAX_DIM, new_weights, 1, m_weights, 1);
}

void Model::update_model_by_add(const double * grad)
{
	vdAdd(MAX_DIM, m_weights, grad, m_weights);
}

#include "LSvm.h"


double LSvm::get_loss(const double* weights ) const
{
	if (weights == nullptr) weights = m_weights;
	double loss = 0;
	// mkl_set_num_threads_local(1);
	// #pragma	omp parallel for num_threads(THREAD_NUM_OMP)
	for (int i = 0; i < N; ++i) {
		double yxw = dot(MAX_DIM, weights, 1, X + MAX_DIM * i, 1)*Y[i];
        if(abs(yxw-1) <= 0.5){
            loss += (1.5-yxw)*(1.5-yxw)/2;
        }else if(yxw < 0.5){
            loss += 1-yxw;
        }
	}
	// mkl_set_num_threads_local(0); 
	loss /= N;
	loss += 0.5*m_lambda[0]*squared_norm(MAX_DIM, weights);
	return loss;
}

void LSvm::get_gradient(int idx, double * grad, const double* weights, int wndim, bool add) 
{
	if (weights == nullptr) weights = m_weights;
	double init_value = (double)add;
	if (wndim == 1) {
        double z = dot(MAX_DIM, weights, 1, X + MAX_DIM * idx, 1);
		double yxw = Y[idx] * z;
		axpby(MAX_DIM, m_lambda[0], weights, 1, init_value, grad, 1);
        if(abs(yxw-1)<=0.5){
            axpby(MAX_DIM, (yxw-1.5)*Y[idx], X + MAX_DIM * idx, 1, 1.0, grad, 1);    
        }else if(yxw<0.5){
            axpby(MAX_DIM, -Y[idx], X + MAX_DIM * idx, 1, 1.0, grad, 1);    
        }
	}else {
		throw "wndim>1";
	}
}

void LSvm::get_gradient_mm(double* data_x, double* data_y, int n_samples, double * grad, const double * weights, int wndim)
{

	if (weights == nullptr) weights = m_weights;
	int len = wndim * n_samples;
	double* yxw = (double*)mkl_malloc(sizeof(double)*len, 64);
	gemm(CblasRowMajor, CblasNoTrans, CblasTrans, wndim, n_samples, MAX_DIM, 1.0, weights, MAX_DIM,
		data_x, MAX_DIM, 0, yxw, n_samples);
	// #pragma	omp parallel for num_threads(5)
	for (int i = 0; i < len; ++i) {
		yxw[i] = yxw[i]*data_y[i%n_samples];
		if(yxw[i]>1.5){
			yxw[i] = 0;
		}else if(yxw[i]>0.5){
			yxw[i] = (yxw[i]-1.5)*data_y[i%n_samples];
		}else{
			yxw[i] = -data_y[i%n_samples];
		}
		// yxw[i] = ma(mi(yxw[i], 0.0),-1.0)*data_y[i%n_samples];
	}
	mem_copy(wndim*MAX_DIM, weights, 1, grad, 1);
	gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, wndim, MAX_DIM, n_samples, 1.0 / n_samples, yxw,
		n_samples, data_x, MAX_DIM, m_lambda[0], grad, MAX_DIM);
	mkl_free(yxw);
}

void LSvm::get_hess_vt(int idx, const double * vt, double * hess_vt, const double * weights, int vndim, bool add) 
{
	if (weights == nullptr) weights = m_weights;
	double yxw = dot(MAX_DIM, weights, 1, X + MAX_DIM * idx, 1)*Y[idx];
	double init_value = (double)add;
	if (vndim == 1) {
		if(yxw>1.5||yxw<0.5) yxw=0;
		else yxw=1;
		double value = yxw * dot(MAX_DIM, vt, 1, X + MAX_DIM * idx, 1);
		axpby(MAX_DIM, m_lambda[0], vt, 1, init_value, hess_vt, 1);
		if(yxw!=0.0) axpby(MAX_DIM, value, X + MAX_DIM * idx, 1, 1.0, hess_vt, 1);
	}else {	
		throw "vndim>1";
	}
}


void LSvm::get_hess_vt_mm(double * data_x, double * data_y, int n_samples, const double * vt, 
	double * hess_vt, const double * weights, int vndim, int _power_iter, double* B)
{
	if (weights == nullptr) weights = m_weights;
	
	double* yxw = (double*)mkl_malloc(sizeof(double)*n_samples, 64);
	double* dx_vt = (double*)mkl_malloc(sizeof(double)*n_samples*vndim, 64);

	// yxw == w@X*Y
	gemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, n_samples, MAX_DIM, 1.0, weights, MAX_DIM,
		data_x, MAX_DIM, 0, yxw, n_samples);
	int count = 0;
	for(int i = 0; i < n_samples; ++i){
		yxw[i] = data_y[i]*yxw[i];
		if(yxw[i]>1.5||yxw[i]<0.5){
			yxw[i] = 0;
		}else{
			 if(i!=count){
				 mem_copy(MAX_DIM, data_x + i * MAX_DIM, 1, data_x + count * MAX_DIM, 1);
			 }
			 yxw[i] = 1, count++;

		}
	}

	// dx_vt=(x@vt)*hess_value
	if(count>0)
	gemm(CblasRowMajor, CblasNoTrans, CblasTrans, vndim, count, MAX_DIM, 1.0, vt, MAX_DIM, data_x, MAX_DIM, 0, dx_vt, count);
	//hess_vt = lambda*vt + 1.0/n_smples *  x @dx_vt
	// mem_copy(MAX_DIM*vndim, vt, 1, hess_vt, 1);
	axpby(MAX_DIM*vndim, m_lambda[0], vt, 1, 0, hess_vt, 1);
	if(count>0)
	gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, vndim, MAX_DIM, count, 1.0/n_samples, dx_vt, count, data_x, MAX_DIM, 1, hess_vt, MAX_DIM);
	// printf("_power_iter%d\n",_power_iter);
	if(~_power_iter){
		//just for lant optimization

		double* tau = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
		//2q power
		for(int i = 0; i < _power_iter; ++i){
			// mem_copy(MAX_DIM*vndim, hess_vt, 1, dx_vt, 1);
			gemm(CblasRowMajor, CblasNoTrans, CblasTrans, vndim, count, MAX_DIM, 1.0, hess_vt, MAX_DIM, data_x, MAX_DIM, 0, dx_vt, count);
			gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, vndim, MAX_DIM, count, 1.0/n_samples, dx_vt, count, data_x, MAX_DIM, m_lambda[0], hess_vt, MAX_DIM);
			gemm(CblasRowMajor, CblasNoTrans, CblasTrans, vndim, count, MAX_DIM, 1.0, hess_vt, MAX_DIM, data_x, MAX_DIM, 0, dx_vt, count);
			gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, vndim, MAX_DIM, count, 1.0/n_samples, dx_vt, count, data_x, MAX_DIM, m_lambda[0], hess_vt, MAX_DIM);
			LAPACKE_dgelqf(CblasRowMajor, vndim, MAX_DIM, hess_vt, MAX_DIM, tau);
			LAPACKE_dorglq(CblasRowMajor, vndim, MAX_DIM, vndim, hess_vt, MAX_DIM, tau);
		}

		//orthogonalization
		LAPACKE_dgelqf(CblasRowMajor, vndim, MAX_DIM, hess_vt, MAX_DIM, tau);
		LAPACKE_dorglq(CblasRowMajor, vndim, MAX_DIM, vndim, hess_vt, MAX_DIM, tau);
	
		// to get B
		// dx_vt = q_matrix@X.T
		// B = q_matrix@(X.T@X/n_samples + lambda@I)@q_matrix.T
		//   = dx_vt@dx_vt.T/n_samples + lambda*q_matrix@q_matrix.T
		gemm(CblasRowMajor, CblasNoTrans, CblasTrans, vndim, count, MAX_DIM, 1.0, hess_vt, MAX_DIM, data_x, MAX_DIM, 0, dx_vt, count);
		gemm(CblasRowMajor, CblasNoTrans, CblasTrans, vndim, vndim, MAX_DIM, m_lambda[0], hess_vt, MAX_DIM, hess_vt, MAX_DIM, 0, B, vndim);
		gemm(CblasRowMajor, CblasNoTrans, CblasTrans, vndim, vndim, count, 1.0/n_samples, dx_vt, count, dx_vt, count, 1.0, B, vndim);
		mkl_free(tau);
	}

	
	mkl_free(dx_vt);
	mkl_free(yxw);
}
void LSvm::get_hessian(const int idx, double * hess, const double * weights, bool add)
{
	if (weights == nullptr) weights = m_weights;
	double yxw = dot(MAX_DIM, weights, 1, X + MAX_DIM * idx, 1)*Y[idx];
	double hess_value = 0;
	if(0.5<yxw&&yxw<1.5) hess_value = 1;
	double init_value = (double)add;
	if (!add) { mem_zero(MAX_DIM*MAX_DIM, hess); }
	if(hess_value>0)
		ger(CblasRowMajor, MAX_DIM, MAX_DIM, hess_value, X + MAX_DIM * idx, 1, X + MAX_DIM * idx, 1, hess, MAX_DIM);
	if (!add) {
		for (int i = 0; i < MAX_DIM; ++i) hess[i*MAX_DIM + i] += m_lambda[0];
	}
}



void LSvm::get_hessian_mm(double * data_x, double * data_y, int n_samples, double * hess, const double * weights)
{
	if (weights == nullptr) weights = m_weights;
	double* yxw = (double*)mkl_malloc(sizeof(double)*n_samples, 64);
	double* dx = (double*)mkl_malloc(sizeof(double)*n_samples*MAX_DIM, 64);
	mem_copy(n_samples*MAX_DIM, data_x, 1, dx, 1);

	// yxw = w@X.T*Y
	gemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, n_samples, MAX_DIM, 1.0, weights, MAX_DIM,
		data_x, MAX_DIM, 0, yxw, n_samples);
	
	for(int i = 0; i < n_samples; ++i){
		yxw[i] = data_y[i]*yxw[i];
		if(yxw[i]>1.5||yxw[i]<0.5){
			yxw[i] = 0;
		}else yxw[i] = 1;
	}
	
	//#pragma	omp parallel for num_threads(THREAD_NUM_OMP)
	for (int i = 0; i < n_samples; ++i) {
		scal(MAX_DIM, yxw[i], dx + i * MAX_DIM, 1);
	}
	//mkl_set_num_threads_local(0);

	//init hessian, hessian = lambda*I
	mem_zero(MAX_DIM*MAX_DIM, hess);
	for (int i = 0; i < MAX_DIM; ++i) {
		hess[i*MAX_DIM + i] = m_lambda[0];
	}
	//hess += 1.0/n_samples* (dx.T@dx)
	gemm(CblasRowMajor, CblasTrans, CblasNoTrans, MAX_DIM, MAX_DIM, n_samples, 1.0 / n_samples, dx, MAX_DIM, dx, MAX_DIM, 1.0, hess, MAX_DIM);

	mkl_free(dx);
	mkl_free(yxw);
}

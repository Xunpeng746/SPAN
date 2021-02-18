#include "Slbfgs.h"





Records Slbfgs::run(int iter_num, Records records, double * weights)
{
	double data_pass = 0;
	MemFactory memFac = MemFactory();
	double* grad_full = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* s_array = (double*)mkl_malloc(sizeof(double)*MAX_DIM*store_num, 64);
	double* y_array = (double*)mkl_malloc(sizeof(double)*MAX_DIM*store_num, 64);
	double* x_array = (double*)mkl_malloc(sizeof(double)*MAX_DIM*update_period, 64);
	double* last_mu = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* grad_last = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* grad_now = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* grad_mu = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* wt = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* w_mean = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* old_w = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* hess_inv_vt = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	int* idx = (int*)mkl_malloc(sizeof(int)*ma(b,bh), 64);
	VSLStreamStatePtr stream_uniform;
	vslNewStream(&stream_uniform, VSL_BRNG_MT19937, __rdtsc());
	
	double* hess_exact = memFac.malloc_double(MAX_DIM*MAX_DIM);
	double* hess_approx = memFac.malloc_double(MAX_DIM*MAX_DIM);
	double* hess_diff = memFac.malloc_double(MAX_DIM*MAX_DIM);

	model->init_model(weights);
	mem_copy(MAX_DIM, model->get_model(), 1, last_mu, 1);
	log_start();
	auto start = std::chrono::high_resolution_clock::now();
	int iter_cnt = 0, t = 0, now_store_num = 0;
	log_iter_end(start, data_pass, records);
	while (iter_cnt < iter_num) {
		data_pass += 1.0;
		iter_cnt += 1;
		model->get_full_gradient(grad_full);
		mem_copy(MAX_DIM, model->get_model(), 1, wt, 1);
		for (int i = 0; i < m; ++i) {
			randint(b, idx, 0, N, stream_uniform);
			model->get_gradient(idx, b, grad_now, wt);
			model->get_gradient(idx, b, grad_last);
			vdSub(MAX_DIM, grad_now, grad_last, grad_mu);
			vdAdd(MAX_DIM, grad_mu, grad_full, grad_mu);

			mem_copy(MAX_DIM, wt, 1, old_w, 1);
			if (now_store_num < store_num) {
				axpy(MAX_DIM, -step_size, grad_mu, 1, wt, 1);
			}else {
				get_hess_inv_vt_approx(s_array, y_array, grad_mu, hess_inv_vt, now_store_num);
				axpy(MAX_DIM, -step_size, hess_inv_vt, 1, wt, 1);
			}
			if (t%update_period == 0 && t > 0) {
				now_store_num += 1;
				int store_idx = now_store_num % store_num;
				//compute mean_x in update_period
				memset(w_mean, 0, sizeof(double)*MAX_DIM);
				for (int j = 0; j < update_period; ++j) {
					vdAdd(MAX_DIM, w_mean, x_array + MAX_DIM * j, w_mean);
				}
				scal(MAX_DIM, 1.0 / update_period, w_mean, 1);
				randint(bh, idx, 0, N, stream_uniform);
				vdSub(MAX_DIM, w_mean, last_mu, s_array + MAX_DIM * store_idx);
				model->get_hess_vt(idx, bh, s_array + MAX_DIM * store_idx, y_array + MAX_DIM * store_idx, wt);
				mem_copy(MAX_DIM, w_mean, 1, last_mu, 1);
			}
			mem_copy(MAX_DIM, old_w, 1, x_array + MAX_DIM * (t%update_period), 1);
			t += 1;
		}

		if (PRINT_HESS) {
			model->get_full_hessian(hess_exact);
			get_hess_inv_approx(s_array, y_array, hess_approx, now_store_num);
			Util::inv(hess_approx, MAX_DIM);
			vdSub(MAX_DIM*MAX_DIM, hess_exact, hess_approx, hess_diff);
			double hess_error = Util::spectral_norm(hess_diff, MAX_DIM);
			records.set_hess_error(hess_error);
		}
		model->update_model(wt);
		log_iter_end(start, data_pass, records);
	}
	log_end(records);
	mkl_free(grad_full);
	mkl_free(s_array);
	mkl_free(y_array);
	mkl_free(x_array);
	mkl_free(last_mu);
	mkl_free(grad_last);
	mkl_free(grad_now);
	mkl_free(grad_mu);
	mkl_free(wt);
	mkl_free(w_mean);
	mkl_free(old_w);
	mkl_free(hess_inv_vt);
	mkl_free(idx);

	vslDeleteStream(&stream_uniform);
	return records;
}



void Slbfgs::get_hess_inv_vt_approx(double * s_array, double * y_array, double* vt, double* res,int now_store_num)
{
	double* rho = (double*)mkl_malloc(sizeof(double)*store_num, 64);
	double* rho_sq = (double*)mkl_malloc(sizeof(double)*store_num, 64);
	mem_copy(MAX_DIM, vt, 1, res, 1);
	int idx = 0;
	double *s = s_array + MAX_DIM * idx, *y = y_array + MAX_DIM * idx;
	for (int j = now_store_num; j > now_store_num - store_num; --j) {
		idx = j% store_num;
		s = s_array + MAX_DIM * idx, y = y_array + MAX_DIM * idx;
		rho[idx] = 1.0 / (dot(MAX_DIM, s, 1, y, 1));
		rho_sq[idx] = rho[idx] * dot(MAX_DIM, s, 1, res, 1);
		axpy(MAX_DIM, -rho_sq[idx], y, 1, res, 1);
	}
	idx = (now_store_num - store_num + 1) % store_num;
	s = s_array + MAX_DIM * idx, y = y_array + MAX_DIM * idx;
	double sy = dot(MAX_DIM, s, 1, y, 1);
	double yy = dot(MAX_DIM, y, 1, y, 1);
	scal(MAX_DIM, sy / yy, res, 1);
	//现在所有从右往左已经迭代到最里层了, 后面就需要左边项不断相乘并加上右边项
	for (int j = now_store_num - store_num + 1; j <= now_store_num; ++j) {
		idx = j % store_num;
		s = s_array + MAX_DIM * idx, y = y_array + MAX_DIM * idx;
		double beta = rho[idx] * dot(MAX_DIM, y, 1, res, 1);
		axpy(MAX_DIM, rho_sq[idx]-beta, s, 1, res, 1);
	}
	mkl_free(rho);
}

void Slbfgs::get_hess_inv_approx(double * s_array, double * y_array, double * res, int now_store_num)
{
	//double* rho = (double*)mkl_malloc(sizeof(double)*store_num, 64);
	double rho;
	double* mat_l = (double*)mkl_malloc(sizeof(double)*MAX_DIM*MAX_DIM, 64);
	double* mat_ll = (double*)mkl_malloc(sizeof(double)*MAX_DIM*MAX_DIM, 64);
	int idx = 0;
	idx = now_store_num % store_num;
	double *s = s_array + MAX_DIM * idx, *y = y_array + MAX_DIM * idx;
	double sy = dot(MAX_DIM, s, 1, y, 1);
	double yy = dot(MAX_DIM, y, 1, y, 1);
	mem_zero(MAX_DIM*MAX_DIM, res);
	for (int i = 0; i < MAX_DIM; ++i) {
		res[i + i * MAX_DIM] += sy / yy;
	}
	for (int j = now_store_num - store_num+1; j<=now_store_num; ++j) {
		idx = j % store_num;
		s = s_array + MAX_DIM * idx, y = y_array + MAX_DIM * idx;
		rho = 1.0 / (dot(MAX_DIM, s, 1, y, 1));
		mem_zero(MAX_DIM*MAX_DIM, mat_l);
		ger(CblasRowMajor, MAX_DIM, MAX_DIM, -rho, s, 1, y, 1, mat_l, MAX_DIM);
		#pragma omp parallel for num_threads(THREAD_NUM_OMP)
		for (int i = 0; i < MAX_DIM; ++i) {
			mat_l[i + i * MAX_DIM] += 1;
		}
		gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MAX_DIM, MAX_DIM, MAX_DIM, 1.0, mat_l, MAX_DIM, res, MAX_DIM, 0, mat_ll, MAX_DIM);
		gemm(CblasRowMajor, CblasNoTrans, CblasTrans, MAX_DIM, MAX_DIM, MAX_DIM, 1.0, mat_ll, MAX_DIM, mat_l, MAX_DIM, 0, res, MAX_DIM);
		ger(CblasRowMajor, MAX_DIM, MAX_DIM, rho, s, 1, s, 1, res, MAX_DIM);
	}
	mkl_free(mat_l);
	mkl_free(mat_ll);
}

Slbfgs::~Slbfgs()
{
	mkl_free(ones);
}

#include "Sbbfgs.h"




Sbbfgs::~Sbbfgs()
{
}

Records Sbbfgs::run(int iter_num, Records records, double * weights)
{
	MemFactory memFac = MemFactory();
	double data_pass = 0;
	double* d_array = (double*)mkl_malloc(sizeof(double)*MAX_DIM*rank*store_num, 64);
	double* y_array = (double*)mkl_malloc(sizeof(double)*MAX_DIM*rank*store_num, 64);
	double* delta_array = (double*)mkl_malloc(sizeof(double)*rank*rank*store_num, 64);
	double* grad_last = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* grad_now = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* grad_mu = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* grad_d = (double*)mkl_malloc(sizeof(double)*rank, 64);
	double* grad_d_delta = (double*)mkl_malloc(sizeof(double)*rank, 64);
	double* wt = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* grad_full = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* hess_inv_vt = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	int* idx = (int*)mkl_malloc(sizeof(int)*ma(b, bh), 64);
	VSLStreamStatePtr stream_uniform;
	vslNewStream(&stream_uniform, VSL_BRNG_MT19937, __rdtsc());

	double* hess_exact = memFac.malloc_double(MAX_DIM*MAX_DIM);
	double* hess_approx = memFac.malloc_double(MAX_DIM*MAX_DIM);
	double* hess_diff = memFac.malloc_double(MAX_DIM*MAX_DIM);
	double* vt_tmp = memFac.malloc_double(MAX_DIM);

	double* d_t = nullptr, *y_t = nullptr, *delta_t=nullptr;
	model->init_model(weights);
	log_start();
	auto start = std::chrono::high_resolution_clock::now();
	int iter_cnt = 0,  t = 0;
	log_iter_end(start, data_pass, records);
	while (iter_cnt < iter_num) {
		iter_cnt += 1;
		model->get_full_gradient(grad_full);
		mem_copy(MAX_DIM, model->get_model(), 1, wt, 1);
		for (int i = 0; i < m; ++i) {
			t += 1;
			//compute svrg
			// data_pass += b*2.0/N;
			randint(b, idx, 0, N, stream_uniform);
			model->get_gradient(idx, b, grad_now, wt);
			model->get_gradient(idx, b, grad_last);
			vdSub(MAX_DIM, grad_now, grad_last, grad_mu);
			vdAdd(MAX_DIM, grad_mu, grad_full, grad_mu);

			//compute Dt, Yt£¬ Delta_t

			randint(bh, idx, 0, N, stream_uniform);
			d_t = d_array + (t%store_num)*MAX_DIM*rank;
			y_t = y_array + (t%store_num)*MAX_DIM*rank;
			delta_t = delta_array + (t%store_num)*rank*rank;
			Util::get_randn(d_t, MAX_DIM*rank, 0, 1);

			// data_pass += bh*rank*1.0/N;
			model->get_hess_vt(idx, bh, d_t, y_t, wt, rank);
			//delta_t = inv(d_t@y_t.T)
			gemm(CblasRowMajor, CblasNoTrans, CblasTrans, rank, rank, MAX_DIM, 1, d_t, MAX_DIM, y_t, MAX_DIM, 0, delta_t, rank);
			Util::inv(delta_t, rank);

			//block lbfgs update
			if (t >= store_num) {
				get_block_bfgs_update(d_array, y_array, delta_array, grad_mu, hess_inv_vt, t);

			}else {
				gemv(CblasRowMajor, CblasNoTrans, rank, MAX_DIM, 1, d_t, MAX_DIM, grad_mu, 1, 0, grad_d, 1);
				gemv(CblasRowMajor, CblasNoTrans, rank, rank, 1.0, delta_t, rank, grad_d, 1, 0, grad_d_delta, 1);
				gemv(CblasRowMajor, CblasTrans, rank, MAX_DIM, 1, d_t, MAX_DIM, grad_d_delta, 1, 0, hess_inv_vt, 1);
			}

			//update
			axpy(MAX_DIM, -step_size, hess_inv_vt, 1, wt, 1);
		}
		data_pass += 1.0 ;
		if (PRINT_HESS) {
			model->get_full_hessian(hess_exact);
			get_hess_inv_approx(d_array, y_array, delta_array, hess_approx, t);
			Util::inv(hess_approx, MAX_DIM);
			vdSub(MAX_DIM*MAX_DIM, hess_exact, hess_approx, hess_diff);
			double hess_error = Util::spectral_norm(hess_diff, MAX_DIM);
			double hess_approx_norm = Util::spectral_norm(hess_approx, MAX_DIM);
			records.set_hess_error(hess_error);
		}
		model->update_model(wt);

		log_iter_end(start, data_pass, records);
	}
	log_end(records);


	mkl_free(d_array);
	mkl_free(y_array);
	mkl_free(delta_array);
	mkl_free(grad_last);
	mkl_free(grad_now);
	mkl_free(grad_mu);
	mkl_free(grad_d);
	mkl_free(grad_d_delta);
	mkl_free(wt);
	mkl_free(grad_full);
	mkl_free(hess_inv_vt);
	mkl_free(idx);
	vslDeleteStream(&stream_uniform);
	return records;
}



void Sbbfgs::get_block_bfgs_update(double * d_array, double * y_array, double * delta_array, double * vt, double * res, int now_store_num)
{
	double* alpha_array = (double*)mkl_malloc(sizeof(double)*store_num*rank, 64);
	double* belta = (double*)mkl_malloc(sizeof(double)*rank, 64);
	double* tmp = (double*)mkl_malloc(sizeof(double)*rank, 64);
	mem_copy(MAX_DIM, vt, 1, res, 1);
	int idx = 0;
	double *d = nullptr, *y = nullptr, *delta = nullptr, *alpha = nullptr;
	for (int i = now_store_num; i > now_store_num - store_num; --i) {
		idx = i % store_num;
		d = d_array + MAX_DIM * rank*idx, y = y_array + MAX_DIM * rank*idx, delta = delta_array + rank * rank*idx;
		alpha = alpha_array + rank * idx;
		gemv(CblasRowMajor, CblasNoTrans, rank, MAX_DIM, 1.0, d, MAX_DIM, res, 1, 0, tmp, 1);
		gemv(CblasRowMajor, CblasNoTrans, rank, rank, 1.0, delta, rank, tmp, 1, 0, alpha, 1);
		gemv(CblasRowMajor, CblasTrans, rank, MAX_DIM, -1, y, MAX_DIM, alpha, 1, 1.0, res, 1);
	}
	for (int i = now_store_num - store_num + 1; i <= now_store_num; ++i) {
		idx = i % store_num;
		d = d_array + MAX_DIM * rank*idx, y = y_array + MAX_DIM * rank*idx, delta = delta_array + rank * rank*idx;
		alpha = alpha_array + rank * idx;

		gemv(CblasRowMajor, CblasNoTrans, rank, MAX_DIM, 1.0, y, MAX_DIM, res, 1, 0, tmp, 1);
		gemv(CblasRowMajor, CblasNoTrans, rank, rank, 1.0, delta, rank, tmp, 1, 0, belta, 1);
		vdSub(rank, alpha, belta, tmp);
		gemv(CblasRowMajor, CblasTrans, rank, MAX_DIM, 1.0, d, MAX_DIM, tmp, 1, 1, res, 1);
	}

	mkl_free(belta);
	mkl_free(tmp);
	mkl_free(alpha_array);
}

void Sbbfgs::get_hess_inv_approx(double * d_array, double * y_array, double * delta_array, double * res, int now_store_num)
{
	double* alpha_array = (double*)mkl_malloc(sizeof(double)*store_num*rank, 64);
	double* mat_l = (double*)mkl_malloc(sizeof(double)*MAX_DIM*MAX_DIM, 64);
	double* mat_ll = (double*)mkl_malloc(sizeof(double)*MAX_DIM*MAX_DIM, 64);
	double* tmp = (double*)mkl_malloc(sizeof(double)*MAX_DIM*rank, 64);

	mem_zero(MAX_DIM*MAX_DIM, res);
	for (int i = 0; i < MAX_DIM; ++i) {
		res[i + i * MAX_DIM] = 1;
	}
	int idx = 0;
	double *d = nullptr, *y = nullptr, *delta = nullptr, *alpha = nullptr;
	for (int i = now_store_num - store_num+1; i <= now_store_num ; ++i) {
		idx = i % store_num;
		d = d_array + MAX_DIM * rank*idx, y = y_array + MAX_DIM * rank*idx, delta = delta_array + rank * rank*idx;
		
		gemm(CblasRowMajor, CblasTrans, CblasNoTrans, MAX_DIM, rank, rank, 1.0, d, MAX_DIM, delta, rank, 0, tmp, rank);
		gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MAX_DIM, MAX_DIM, rank, -1.0, tmp, rank, y, MAX_DIM, 0, mat_l, MAX_DIM);
		#pragma omp parallel for num_threads(THREAD_NUM_OMP)
		for (int i = 0; i < MAX_DIM; ++i) {
			mat_l[i + i * MAX_DIM] += 1;
		}
		gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MAX_DIM, MAX_DIM, MAX_DIM, 1.0, mat_l, MAX_DIM, res, MAX_DIM, 0, mat_ll, MAX_DIM);
		gemm(CblasRowMajor, CblasNoTrans, CblasTrans, MAX_DIM, MAX_DIM, MAX_DIM, 1.0, mat_ll, MAX_DIM, mat_l, MAX_DIM, 0, res, MAX_DIM);

		gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MAX_DIM, MAX_DIM, rank, 1.0, tmp, rank, d, MAX_DIM, 1.0, res, MAX_DIM);
	}
	mkl_free(mat_l);
	mkl_free(mat_ll);
	mkl_free(tmp);
	mkl_free(alpha_array);

}

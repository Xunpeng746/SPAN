#include "Svrg.h"



Svrg::~Svrg()
{
}

Records Svrg::run(int iter_num, Records records, double * weights)
{
	double data_pass = 0;
	double* grad_full = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* grad_last = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* grad_now = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* wt = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* grad_mu = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	int* idx = (int*)mkl_malloc(sizeof(int)*m, 64);
	int iter_cnt = 0;
	log_start();
	auto start = std::chrono::high_resolution_clock::now();
	model->init_model(weights);
	log_iter_end(start, data_pass, records);
	while (iter_cnt < iter_num) {
		iter_cnt += 1;
		Util::get_rand_choice(idx, m, 0, N);
		mem_copy(MAX_DIM, model->get_model(), 1, wt, 1);
		model->get_full_gradient(grad_full);
		for (int i = 0; i < m; ++i) {
			model->get_gradient(idx[i], grad_now, wt);
			model->get_gradient(idx[i], grad_last);
			vdSub(MAX_DIM, grad_now, grad_last, grad_mu);
			vdAdd(MAX_DIM, grad_mu, grad_full, grad_mu);
			axpy(MAX_DIM, -step_size, grad_mu, 1, wt, 1);
		}
		model->update_model(wt);

		data_pass += 1.0 ;
		log_iter_end(start, data_pass, records);
	}
	log_end(records);
	return records;
}

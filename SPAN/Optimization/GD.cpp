#include "GD.h"





Records GD::run(int iter_num, Records records, double * weights)
{
	double iter_cnt = 0;
	double data_pass = 0;
	double loss = 0.0;
	double* full_grad = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	printf("step_size:%.3f\n", step_size);
	model->init_model();
	auto start = std::chrono::high_resolution_clock::now();
	log_start();
	while (iter_cnt < iter_num) {
		iter_cnt += 1;
		get_full_gradient(full_grad);
		axpy(MAX_DIM, -step_size, full_grad, 1, model->get_model(), 1);
		data_pass += 1.0;
		log_iter_end(start, data_pass, records);
	}
	log_end(records);
	mkl_free(full_grad);
	return Records();
}

GD::~GD()
{
}

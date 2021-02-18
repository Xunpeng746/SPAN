#include "NewSamp.h"

NewSamp::~NewSamp()
{
}

Records NewSamp::run(int iter_num, Records records, double * weights)
{
	double data_pass = 0;
	double hess_error = 1;
	MemFactory memFac = MemFactory();
	double* hess = memFac.malloc_double(MAX_DIM*MAX_DIM);
	double* u = memFac.malloc_double(MAX_DIM*rank);
	double* uts = memFac.malloc_double(MAX_DIM*rank);
	double* grad_u = memFac.malloc_double(rank);
	double* s = memFac.malloc_double(MAX_DIM);
	double* s_tmp = memFac.malloc_double(MAX_DIM);
	double* grad =memFac.malloc_double(MAX_DIM);
	double* hess_inv_vt = memFac.malloc_double(MAX_DIM);
	int* idx = memFac.malloc_int(sample_size);

	double* hess_exact = memFac.malloc_double(MAX_DIM*MAX_DIM);
	double* hess_approx = memFac.malloc_double(MAX_DIM*MAX_DIM);
	double* hess_diff = memFac.malloc_double(MAX_DIM*MAX_DIM);

	int iter_cnt = 0;
	model->init_model(weights); 
	log_start();
	auto start = std::chrono::high_resolution_clock::now();
	log_iter_end(start, data_pass, records);
	while(iter_cnt < iter_num){
		iter_cnt += 1;
		model->get_full_gradient(grad);
		Util::get_rand_choice(idx, sample_size, 0, N);
		model->get_hessian(idx, sample_size, hess);
		mkl_set_num_threads_local(1);
		Util::ksvd(hess, u, s, MAX_DIM, rank);
		mkl_set_num_threads_local(0);
		if(PRINT_HESS){
		 	model->get_full_hessian(hess_exact);
		 	mkl_domatcopy('R', 'T', MAX_DIM, rank, 1.0, u, rank, uts, MAX_DIM);
			double lambda_const = s[0]*1.2;
		 	for(int i = 0; i < rank; ++i){
		 		scal(MAX_DIM, s[i] - lambda_const, uts+i*MAX_DIM, 1);
		 	}
		 	gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MAX_DIM, MAX_DIM, rank,1.0,
		 	 u, rank, uts, MAX_DIM, 0, hess_approx, MAX_DIM);
			for (int i = 0; i < MAX_DIM; ++i) hess_approx[i*MAX_DIM + i] += lambda_const;

		 	vdSub(MAX_DIM*MAX_DIM, hess_exact, hess_approx, hess_diff);
			hess_error = Util::spectral_norm(hess_diff, MAX_DIM);
			records.set_hess_error(hess_error);
		 }
		double lambda_const = 1 / s[0];
		vdLinearFrac(rank, s, s, -lambda_const, 1, 1.0, 0.0, s_tmp);
		gemv(CblasRowMajor, CblasTrans, MAX_DIM, rank, 1.0, u, rank, grad, 1, 0.0, grad_u, 1);
		vdMul(rank, grad_u, s_tmp, grad_u);
		gemv(CblasRowMajor, CblasNoTrans, MAX_DIM, rank, 1.0, u, rank, grad_u, 1, 0, hess_inv_vt, 1);
		axpy(MAX_DIM, lambda_const, grad, 1, hess_inv_vt, 1);

		model->step(step_size, hess_inv_vt);
		data_pass += 1.0;


		log_iter_end(start, data_pass, records);
	}
	log_end(records);

	return records;
}

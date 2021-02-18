#include "Lissa.h"
// extern size_t MAX_DIM;



Lissa::~Lissa()
{
}

Records Lissa::run(int iter_num, Records records, double * weights)
{
	double iter_cnt = 0;
	double data_pass = 0, hess_error = 0;
	MemFactory memFac = MemFactory();
	double* grad = memFac.malloc_double(MAX_DIM);
	double* vt = memFac.malloc_double(MAX_DIM);
	double* u = memFac.malloc_double(MAX_DIM);
	double* u_ave = memFac.malloc_double(MAX_DIM);
	int* idx = memFac.malloc_int(s2);
	int* idx_set = memFac.malloc_int(t1);

	double* hess_exact = memFac.malloc_double(MAX_DIM*MAX_DIM);
	double* hess_approx = memFac.malloc_double(MAX_DIM*MAX_DIM);
	double* hess_diff = memFac.malloc_double(MAX_DIM*MAX_DIM);
	double* hess_single = memFac.malloc_double(MAX_DIM*MAX_DIM);
	double* hess_test = memFac.malloc_double(MAX_DIM*MAX_DIM);
	double* hess_tmp = memFac.malloc_double(MAX_DIM*MAX_DIM);
	double* tmpv = memFac.malloc_double(MAX_DIM);
	Util::get_rand_choice(idx, s2, 0, N);
	model->init_model(weights);
	log_start();
	auto start = std::chrono::high_resolution_clock::now();
	log_iter_end(start, data_pass, records);
	while (iter_cnt < iter_num) {
		iter_cnt += 1;
		data_pass += 1;

		
		memset(grad, 0, sizeof(double)*MAX_DIM);
		memset(u, 0, sizeof(double)*MAX_DIM);
		memset(vt, 0, sizeof(double)*MAX_DIM);
		memset(u_ave, 0, sizeof(double)*MAX_DIM);
		get_full_gradient(grad);
		Util::get_rand_choice(idx_set, t1, 0, N);
		for(int i = 0; i < s1; ++i){
			mem_copy(MAX_DIM, grad, 1, u, 1);
			Util::get_rand_choice(idx, s2, 0, N);
			for (int j = 0; j < s2; ++j) {
				// hess_vt = hessian*vt
				get_hess_vt(idx[j], u, vt);
				// get_hess_vt(idx_set[idx[j]], u, vt);
				axpy(MAX_DIM, -1, vt, 1, u, 1);
				axpy(MAX_DIM, 1, grad, 1, u, 1);
				// vt = (grad + vt - hessian_vt)
			}
			axpy(MAX_DIM, 1.0/s1, u, 1, u_ave, 1);
		}


		if (PRINT_HESS) {
			model->get_full_hessian(hess_exact);
			mem_zero(MAX_DIM*MAX_DIM, hess_approx);
			for (int i = 0; i < MAX_DIM; ++i) hess_approx[i*MAX_DIM + i] = 1;
			for (int i = 0; i < s2; ++i) {
				double *data_x, data_y;
				double hess_value = get_hess_value(idx[i], data_x);
				gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, MAX_DIM, MAX_DIM, 1.0, data_x, MAX_DIM, hess_approx, MAX_DIM, 0, tmpv, MAX_DIM);
				gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MAX_DIM, MAX_DIM, 1, hess_value, data_x, 1, tmpv, MAX_DIM, 0, hess_tmp, MAX_DIM);
				axpy(MAX_DIM*MAX_DIM, model->m_lambda[0], hess_approx, 1, hess_tmp, 1);
				axpy(MAX_DIM*MAX_DIM, -1, hess_tmp, 1, hess_approx, 1);
				for (int j = 0; j < MAX_DIM; ++j) {
					hess_approx[j*MAX_DIM + j] += 1;
				}
			}
			Util::inv(hess_approx, MAX_DIM);
			vdSub(MAX_DIM*MAX_DIM, hess_exact, hess_approx, hess_diff);
			hess_error = Util::spectral_norm(hess_diff, MAX_DIM);
			records.set_hess_error(hess_error);
		}
		step(step_size, u_ave);
		log_iter_end(start, data_pass, records);
	}
	log_end(records);

	return records;
}

double Lissa::get_hess_value(int idx, double *& data_x)
{
	double* weights = model->get_model();
	data_x = model->X + MAX_DIM * idx;
	double yxw = dot(MAX_DIM, weights, 1, data_x, 1)*model->Y[idx];
	double sigmoid = expit(yxw);
	double hess_value = sigmoid * (1 - sigmoid);
	return hess_value;

}

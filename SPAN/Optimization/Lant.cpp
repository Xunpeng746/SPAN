#include "Lant.h"
// extern size_t MAX_DIM;


Records Lant::run(int iter_num, Records records, double * weights)
{

	
	int iter_cnt = 0;
	double data_pass = 0, hess_error = 1;
	int rank = keep_rank + more_rank;
	MemFactory memFac = MemFactory();
	double* full_grad = memFac.malloc_double(MAX_DIM);
	double* hess_inv_vt = memFac.malloc_double(MAX_DIM);
	double* u = memFac.malloc_double(MAX_DIM*rank);
	double* s = memFac.malloc_double(rank);
	double* s_tmp = memFac.malloc_double(rank);
	double* grad_u = memFac.malloc_double(rank);

	double* u_tmp = memFac.malloc_double(MAX_DIM*rank);
	double* hess_exact = memFac.malloc_double(MAX_DIM*MAX_DIM);
	

	double lambda_const = 1.0;
	double* s_keep = s + more_rank;
	double* hess_approx = memFac.malloc_double(MAX_DIM*MAX_DIM);
	int* idx = memFac.malloc_int(sample_size);
	double* hess_diff = memFac.malloc_double(MAX_DIM*MAX_DIM);
	if (weights == nullptr) model->init_model();
	else model->update_model(weights);
	log_start();
	auto start = std::chrono::high_resolution_clock::now();
	log_iter_end(start, data_pass, records);
	while (iter_cnt < iter_num) {
		iter_cnt += 1;
		Util::get_rand_choice(idx, sample_size, 0, N);
		get_hess_approx(u, s, idx);

		if (PRINT_HESS) {
			Util::get_randn(hess_diff, MAX_DIM*MAX_DIM, 0, 1);
			model->get_full_hessian(hess_exact);
			mem_copy(MAX_DIM*rank, u, 1, u_tmp, 1);
			double lambda_const = s[more_rank];
			for (int i = 0; i < rank; ++i) {
				scal(MAX_DIM, s[i] - lambda_const, u_tmp + i * MAX_DIM, 1);
			}
			gemm(CblasRowMajor, CblasTrans, CblasNoTrans, MAX_DIM, MAX_DIM, keep_rank, 1.0, u_tmp + more_rank * MAX_DIM, MAX_DIM,
			u + more_rank * MAX_DIM, MAX_DIM, 0, hess_approx, MAX_DIM);
			for (int i = 0; i < MAX_DIM; ++i) hess_approx[i*MAX_DIM + i] += lambda_const;
			double diff_sum = cblas_dasum(MAX_DIM*MAX_DIM, hess_diff, 1);
			vdSub(MAX_DIM*MAX_DIM,hess_approx, hess_exact, hess_diff);
			hess_error = Util::spectral_norm(hess_diff, MAX_DIM);
			int k = 0;
			records.set_hess_error(hess_error);
		}
			
		int const_num = more_rank;
		model->get_full_gradient(full_grad);
		lambda_const = 2.0 / (s[const_num] + s[const_num + 1]);
		vdLinearFrac(keep_rank, s_keep, s_keep, -lambda_const, 1, 1.0, 0.0, s_tmp);
		gemv(CblasRowMajor, CblasNoTrans, keep_rank, MAX_DIM, 1.0, u + more_rank * MAX_DIM, MAX_DIM, full_grad, 1, 0.0, grad_u, 1);
		vdMul(keep_rank, grad_u, s_tmp, grad_u);
		gemv(CblasRowMajor, CblasTrans, keep_rank, MAX_DIM, 1.0, u + more_rank * MAX_DIM, MAX_DIM, grad_u, 1, 0, hess_inv_vt, 1);
		axpy(MAX_DIM, lambda_const, full_grad, 1, hess_inv_vt, 1);
		axpy(MAX_DIM, -step_size, hess_inv_vt, 1, model->get_model(), 1);
		data_pass+=1;	
		double loss = log_iter_end(start, data_pass, records);
		if(loss<0||loss>2){
			break;
		}
	}
	log_end(records);
	return records;
}



void Lant::
run_stage_a_4_3(const double * rand_matrix, const int * idx, int n_samples, const double * grad, double * q_matrix, double * y_matrix, const double * weights,int _power_iter)
{
	int rank = more_rank + keep_rank;
	double* tau = (double*)mkl_malloc(sizeof(double)*rank, 64);
	double* hess_dot_matrix[2];
	hess_dot_matrix[0] = (double*)mkl_malloc(sizeof(double)*MAX_DIM*rank, 64);
	hess_dot_matrix[1] = q_matrix;
	get_hess_dot_matrix_approx(rand_matrix, hess_dot_matrix[1], idx, n_samples, grad, rank, nullptr, _power_iter);
	// for (int i = 0; i < _power_iter; ++i) {
	// 	get_hess_dot_matrix_approx(hess_dot_matrix[1], hess_dot_matrix[0], idx, n_samples, grad, rank);
	// 	// LAPACKE_dgelqf(CblasRowMajor, rank, MAX_DIM, hess_dot_matrix[0], MAX_DIM, tau);
	// 	// LAPACKE_dorglq(CblasRowMajor, rank, MAX_DIM, rank, hess_dot_matrix[0], MAX_DIM, tau);
	// 	get_hess_dot_matrix_approx(hess_dot_matrix[0], hess_dot_matrix[1], idx, n_samples, grad, rank);
	// 	// LAPACKE_dgelqf(CblasRowMajor, rank, MAX_DIM, hess_dot_matrix[1], MAX_DIM, tau);
	// 	// LAPACKE_dorglq(CblasRowMajor, rank, MAX_DIM, rank, hess_dot_matrix[1], MAX_DIM, tau);
		
	// }
	LAPACKE_dgelqf(CblasRowMajor, rank, MAX_DIM, hess_dot_matrix[1], MAX_DIM, tau);
	LAPACKE_dorglq(CblasRowMajor, rank, MAX_DIM, rank, hess_dot_matrix[1], MAX_DIM, tau);
	if (hess_dot_matrix[1] != q_matrix) {
		mem_copy(MAX_DIM*rank, hess_dot_matrix[1], 1, q_matrix, 1);
	}


	mkl_free(tau);
	mkl_free(hess_dot_matrix[0]);
}

void Lant::run_stage_ab(const double* rand_matrix, int * idx, int n_samples, double* u, double* s, double* weight){
	int rank = more_rank + keep_rank;
	double* q_matrix = (double*)mkl_malloc(sizeof(double)*rank*MAX_DIM, 64);
	double* B = (double*)mkl_malloc(sizeof(double)*rank*rank, 64);
	model->get_hess_vt(idx, n_samples, rand_matrix, q_matrix, weight, rank, power_iter, B);

	// gemm(CblasRowMajor, CblasNoTrans, CblasTrans, rank, rank, MAX_DIM, 1.0, hess_q_matrix, MAX_DIM, q_matrix, MAX_DIM, 
	// 	0, B, rank);
	LAPACKE_dsyevd(CblasRowMajor, 'V', 'U', rank, B, rank, s);
	gemm(CblasRowMajor, CblasTrans, CblasNoTrans, rank, MAX_DIM, rank, 1.0, B, rank, q_matrix, MAX_DIM, 0, u, MAX_DIM);
	mkl_free(B);
	mkl_free(q_matrix);
}

void Lant::
run_stage_b_5_3(const double * q_matrix, int * idx, int n_samples, const double * grad, double * u, double * s, const double * weights)
{
	int rank = more_rank + keep_rank;
	double* hess_q_matrix = u;
	double*  B = (double*)mkl_malloc(sizeof(double)*rank*rank, 64);
	//double*  B_tmp = (double*)mkl_malloc(sizeof(double)*rank*rank, 64);
	get_hess_dot_matrix_approx(q_matrix, hess_q_matrix, idx, n_samples, grad, rank, weights);
	gemm(CblasRowMajor, CblasNoTrans, CblasTrans, rank, rank, MAX_DIM, 1.0, hess_q_matrix, MAX_DIM, q_matrix, MAX_DIM, 
		0, B, rank);
	LAPACKE_dsyevd(CblasRowMajor, 'V', 'U', rank, B, rank, s);
	gemm(CblasRowMajor, CblasTrans, CblasNoTrans, rank, MAX_DIM, rank, 1.0, B, rank, q_matrix, MAX_DIM, 0, u, MAX_DIM);

	mkl_free(B);
	//mkl_free(B_tmp);
}

void Lant::
get_hess_dot_matrix_approx(const double * matrix, double * grad_diff, const int idx[], int n_samples,const double * grad, int ndim, const double * weights, int _power_iter)
{
	if (!use_hess_vt) {
		if (weights == nullptr) weights = model->get_model();
		double* new_weights = (double*)mkl_malloc(sizeof(double)*MAX_DIM*ndim, 64);
		axpby(MAX_DIM*ndim, scale_w, matrix, 1, 0, new_weights, 1);
		cblas_dger(CblasRowMajor, ndim, MAX_DIM, 1.0, one_dim, 1, weights, 1, new_weights, MAX_DIM);
		//out = new_grad-grad
		model->get_gradient(idx, n_samples, grad_diff, new_weights, ndim);
		cblas_dger(CblasRowMajor, ndim, MAX_DIM, -1.0, one_dim, 1, grad, 1, grad_diff, MAX_DIM);
		// recover scale_w, because matrix scale by scalw_w
		scal(MAX_DIM*ndim, 1.0 / scale_w, grad_diff, 1);
		mkl_free(new_weights);
	}else {
		if (weights == nullptr) weights = model->get_model();
		// for (int i = 0; i < ndim; ++i) {
		// model->get_hess_vt(idx, n_samples, matrix, grad_diff, weights, ndim);
		// }
		model->get_hess_vt(idx, n_samples, matrix, grad_diff, weights, ndim,_power_iter);
	}
	
}

void Lant::get_hess_approx(double * u, double * s, int* idx, double * weight)
{
	int rank = more_rank + keep_rank;
	double* q_matrix = (double*)mkl_malloc(sizeof(double)*rank*MAX_DIM, 64);
	double* mini_grad = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* rand_matrix = Util::get_randn(rank*MAX_DIM, 0, 1);
	//idx = Util::get_rand_choice(sample_size, 0, N);
	//Util::get_rand_choice(idx, sample_size, 0, N);
	if(use_hess_vt){
		run_stage_ab(rand_matrix, idx, sample_size, u, s);
	}else{
		model->get_gradient(idx, sample_size, mini_grad);
		// auto st = std::chrono::high_resolution_clock::now();
		if (stage_a==3) {
			run_stage_a_4_3(rand_matrix, idx, sample_size, mini_grad, q_matrix, nullptr);
		}
		// auto ed1 = std::chrono::high_resolution_clock::now();
		if (stage_b==3) {
			run_stage_b_5_3(q_matrix, idx, sample_size, mini_grad, u, s);
		}
	}
	// auto ed2 = std::chrono::high_resolution_clock::now();
	// printf("stageA cost time:%.5f\n", sub(ed1, st));
	// printf("stageB cost time:%.5f\n", sub(ed2, ed1));

	//mkl_free(y_matrix);
	mkl_free(q_matrix);
	mkl_free(mini_grad);
	mkl_free(rand_matrix);
	//mkl_free(idx);
}


void Lant::get_hess_inv_vt_by_usv(double * u, double * s, double * vt, double* hess_inv_vt)
{
	if (vt != hess_inv_vt) {
		mem_copy(MAX_DIM, vt, 1, hess_inv_vt, 1);
	}
	int const_num = more_rank;
	double* s_keep = s + more_rank;
	// auto grad_ed = std::chrono::high_resolution_clock::now();
	double lambda_const = 2.0 / (s[const_num] + s[const_num + 1]);
	double* tmp = (double*)mkl_malloc(sizeof(double)*MAX_DIM * 2, 64);
	double *s_tmp = tmp, *grad_u = tmp + MAX_DIM;

	// hess_inv_vt = grad@u[:,more_rank:]*(1/s[drop:] - lambda_cost)@v[more_rank:,:]
	vdLinearFrac(keep_rank, s_keep, s_keep, -lambda_const, 1, 1.0, 0.0, s_tmp);
	gemv(CblasRowMajor, CblasNoTrans, keep_rank, MAX_DIM, 1.0, u + more_rank * MAX_DIM, MAX_DIM, vt, 1, 0.0, grad_u, 1);
	vdMul(keep_rank, grad_u, s_tmp, grad_u);
	gemv(CblasRowMajor, CblasTrans, keep_rank, MAX_DIM, 1.0, u + more_rank * MAX_DIM, MAX_DIM, grad_u, 1, lambda_const, hess_inv_vt, 1);
	//axpy(MAX_DIM, lambda_const, vt, 1, hess_inv_vt, 1);
	mkl_free(tmp);
}



Lant::~Lant()
{
	mkl_free(one_dim);
}

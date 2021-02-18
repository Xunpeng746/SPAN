#include "TestUtil.h"

void Test::test_norm()
{
	const int N = 3;
	//double* a = (double*)mkl_malloc(sizeof(double)*N*N, 64);
	//mem_zero(N*N, a);
	//for (int i = 0; i < N*N; ++i) a[i] = i;
	double a[9] = { 1,4,9,4, 25, 36, 9, 36, 81 };
	Util::print_matrix("a", a, N, N, N);
	double nrm = Util::spectral_norm(a, N);
	printf("the spectral_norm of a is%.10f\n", nrm);
}

void Test::test_get_hessian_mm()
{
	printf("---------------------test_get_hessian_mm-----------------------\n");
	double *X = nullptr, *Y = nullptr;
	int N = Util::loadMnist49(&X, &Y);
	int M = 2;
	int vndim = 1;
	Model* model = new LSvm(X, Y, N, 1e-4);
	double* hess_exact = (double*)mkl_malloc(sizeof(double)*MAX_DIM*MAX_DIM, 64);
	double* hess_mm = (double*)mkl_malloc(sizeof(double)*MAX_DIM*MAX_DIM, 64);
	double* hess_diff = (double*)mkl_malloc(sizeof(double)*MAX_DIM*MAX_DIM, 64);
	int* idx = (int*)mkl_malloc(sizeof(int)*M, 64);
	for (int i = 0; i < M; ++i) idx[i] = i;
	model->init_model();
	model->get_hessian(idx, M, hess_exact);
	model->get_hessian_mm(X, Y, M, hess_mm);
	vdSub(MAX_DIM*MAX_DIM, hess_exact, hess_mm, hess_diff);
	double hess_exact_norm = squared_norm(MAX_DIM*MAX_DIM, hess_exact);
	double hess_mm_norm = squared_norm(MAX_DIM*MAX_DIM, hess_mm);
	double error = squared_norm(MAX_DIM*MAX_DIM, hess_diff);
	printf("hess_exact_norm is :%.10f\n", hess_exact_norm);
	printf("hess_mm_norm is :%.10f\n", hess_mm_norm);
	printf("hess_diff_norm is :%.10f\n", error);

	mkl_free(X);
	mkl_free(Y);
	mkl_free(hess_exact);
	mkl_free(hess_mm);
	mkl_free(hess_diff);
	mkl_free(idx);
	delete model;
}

void Test::test_get_hess_vt_mm()
{
	printf("---------------------test_get_hess_vt_mm-----------------------\n");
	double *X = nullptr, *Y = nullptr;
	int N = Util::loadMnist49(&X, &Y);
	int M = 2;
	Model* model = new LSvm(X, Y, N, 1e-4); 
	int vndim = 1;
	double* vt = Util::get_randn(MAX_DIM*vndim, 0, 1);
	double* hess_vt_exact = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* hess_vt_mm = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* hess_vt_diff = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	int* idx = (int*)mkl_malloc(sizeof(int)*M, 64);
	for (int i = 0; i < M; ++i) idx[i] = i;
	model->init_model();
	model->get_hess_vt(idx, M, vt, hess_vt_exact);
	model->get_hess_vt_mm(X, Y, M, vt, hess_vt_mm);
	vdSub(MAX_DIM, hess_vt_exact, hess_vt_mm, hess_vt_diff);
	double hess_vt_exact_norm = squared_norm(MAX_DIM, hess_vt_exact);
	double hess_vt_mm_norm = squared_norm(MAX_DIM, hess_vt_mm);
	double error = squared_norm(MAX_DIM, hess_vt_diff);
	printf("hess_vt_exact_norm is :%.10f\n", hess_vt_exact_norm);
	printf("hess_vt_mm_norm is :%.10f\n", hess_vt_mm_norm);
	printf("hess_vt_diff_norm is :%.10f\n", error);

	mkl_free(X);
	mkl_free(Y);
	mkl_free(vt);
	mkl_free(hess_vt_exact);
	mkl_free(hess_vt_mm);
	mkl_free(hess_vt_diff);
	mkl_free(idx);
	delete model;
}

void Test::test_get_gradient_mm()
{
	printf("---------------------test_get_gradient_mm-----------------------\n");
	double *X = nullptr, *Y = nullptr;
	int N = Util::loadMnist49(&X, &Y);
	int M = 2;
	Model* model = new LSvm(X, Y, N, 1e-4);
	double* grad_exact = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* grad_mm = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* grad_diff = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	int* idx = (int*)mkl_malloc(sizeof(int)*M, 64);
	for (int i = 0; i < M; ++i) idx[i] = i;
	model->init_model();
	model->get_gradient(idx, M, grad_exact);
	model->get_gradient_mm(X+0, Y+0, M, grad_mm);
	vdSub(MAX_DIM, grad_exact, grad_mm, grad_diff);
	double grad_exact_norm = squared_norm(MAX_DIM, grad_exact);
	double grad_mm_norm = squared_norm(MAX_DIM, grad_mm);
	double error = squared_norm(MAX_DIM, grad_diff);
	printf("grad_exact_norm is :%.10f\n", grad_exact_norm);
	printf("grad_mm_norm is :%.10f\n", grad_mm_norm);
	printf("grad_diff_norm is :%.10f\n", error);


	mkl_free(X);
	mkl_free(Y);
	mkl_free(grad_exact);
	mkl_free(grad_mm);
	mkl_free(grad_diff);
	mkl_free(idx);
	delete model;
}

void Test::test_logistic_get_gradient(double* X, double* Y, int N) {
	printf("---------------------test_logistic_get_gradient-----------------------\n");
	Model* model = new LSvm(X, Y, N, 1e-4);
	double* grad = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* delta_w = Util::get_randn(MAX_DIM, 0, 1e-7);
	double loss = 0, new_loss = 0, diff_loss = 0, diff_loss_apporx = 0;
	model->init_model();
	loss = model->get_loss();
	model->update_model_by_add(delta_w);
	new_loss = model->get_loss();
	model->get_full_gradient(grad);

	diff_loss = new_loss - loss;
	diff_loss_apporx = dot(MAX_DIM, grad, 1, delta_w, 1);
	double error = diff_loss - diff_loss_apporx;
	double relative_error = error / mi(diff_loss, diff_loss_apporx);
	printf("diff loss:%.10f\n", diff_loss);
	printf("diff loss approx :%.10f\n", diff_loss_apporx);
	printf("error:%.10f\n", error);
	printf("relative_error:%.10f\n", relative_error);
	delete model;
	mkl_free(delta_w);
	mkl_free(grad);
}


void Test::test_mkl_gemm(int N)
{
	std::cout << "start test test_mkl_gemm" << std::endl;
	//const int N = 3;
	const int len = N * N;
	double* a = (double*)mkl_malloc(sizeof(double)*len, 64);
	double* b = (double*)mkl_malloc(sizeof(double)*len, 64);
	for (int i = 0; i < len; ++i) a[i] = i;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, a, N, a, N, 0, b, N);
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("%.3f ", a[i * N + j]);
		}
		printf("\n");
	}
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("%.3f ", b[i*N+j]);
		}
		printf("\n");
	}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, a, N, a, N, 0, a, N);
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("%.3f ", a[i * N + j]);
		}
		printf("\n");
	}
	mkl_free(a);
	mkl_free(b);
	std::cout << "start test test_mkl_gemm" << std::endl;
}

void Test::test_get_randn(int N)
{
	std::cout << "start test test_get_randn" << std::endl;
	//const int N = 1000;
	double* data = Util::get_randn(N, 0, 1);
	for (int i = 0; i < N; ++i) {
		std::cout << i << ":" << data[i] << std::endl;
	}
	
	std::cout << "end test test_get_randn" << std::endl;
	mkl_free(data);
}

void Test::test_get_rand_choice(int N)
{
	std::cout << "start test test_get_rand_choice" << std::endl;
	//const int N = 1000;
	int* data = Util::get_rand_choice(N, 0, N);
	for (int i = 0; i < N; ++i) {
		std::cout << i << ":" << data[i] << std::endl;
	}
	mkl_free(data);
	std::cout << "end test test_get_rand_choice" << std::endl;
}

void Test::test_init_randn(int N)
{
	std::cout << "start test test_init_randn" << std::endl;
	//const int N = 1000;
	double* data = (double*)mkl_malloc(sizeof(double)*N, 64);
	Util::get_randn(data, N, 0, 1);
	for (int i = 0; i < N; ++i) {
		std::cout << i << ":" << data[i] << std::endl;
	}
	mkl_free(data);
	std::cout << "end test test_init_randn" << std::endl;
}

void Test::test_mkl_lq(int M, int N)
{
	int len = M * N;
	double* a = (double*)mkl_malloc(sizeof(double)*M*N, 64);
	double* tau = (double*)mkl_malloc(sizeof(double)*ma(M, N), 64);
	for (int i = 0; i < len; ++i) a[i] = i * i;
	Util::print_matrix("origin matrix",a, M, N, N);
	/*array([[0., -0.50263366, 0.79565928],
		[-0.0531494, -0.60429628, -0.05856375],
		[-0.2125976, -0.50717724, -0.42812124],
		[-0.4783446, -0.21127652, -0.31301317],
		[-0.85039041, 0.28340587, 0.28676045]])*/
	LAPACKE_dgelqf(CblasRowMajor, M, N, a, N, tau);
	LAPACKE_dorglq(CblasRowMajor, M, N, M, a, N, tau);
	Util::print_matrix("q_matrix", a, M, N, N);
	mkl_free(a);
	mkl_free(tau);

}

void Test::test_mkl_svd(int N)
{
	//A = u@s@u.T=array([[0, 1, 4],
	//	[1, 16, 25],
	//	[4, 25, 64]], dtype = int32)
	// s=>array([-0.29575043,  5.41886168, 74.87688876])
	 //u=>array([[-0.03733134, -0.44540245, -0.89455186],
	//[-0.32875167, -0.83987333, 0.43189713],
	//	[-0.94367829, 0.31020871, -0.1150732]])
	double* a = (double*)mkl_malloc(sizeof(double)*N*N, 64);
	double* u = (double*)mkl_malloc(sizeof(double)*N*N, 64);
	double* v = (double*)mkl_malloc(sizeof(double)*N*N, 64);
	double* s = (double*)mkl_malloc(sizeof(double)*N, 64);
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			if(j>=i)	a[i*N + j] = (i*N + j)*(i*N + j);
			else a[i*N + j] = (j*N + i)*(j*N + i);
		}
	}
	mem_copy(N*N, a, 1, u, 1);
	Util::print_matrix("origin matrix", a, N, N, N);
	LAPACKE_dsyevd(CblasRowMajor, 'V', 'U', N, u, N, s);
	mem_copy(N*N, u, 1, v, 1);
	Util::print_matrix("eigen value", s, 1, N, N);
	Util::print_matrix("U matrix", u, N, N, N);
	for (int i = 0; i < N; ++i) {
		scal(N, s[i], u+i, N);
	}
	Util::print_matrix("scale U matrix", u, N, N, N);
	gemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, u, N, v, N, 0, a, N);
	Util::print_matrix("U@S@U.T", a, N, N, N);

	transpose('R', 'T', N, N, 1.0, u, N, N);
	Util::print_matrix("transpose scale U matrix", u, N, N, N);
	mkl_free(a);
	mkl_free(s);
	mkl_free(v);
	mkl_free(u);

}

void Test::test_logistic_hessian(double* X, double* Y, int N, int sample_size)
{
	printf("---------------------test_logistic_hessian-----------------------\n");
	double hess_vt_origin_norm = 0, hess_vt_norm = 0, error = 0, relative_error = 0;
	double* hess = (double*)mkl_malloc(sizeof(double)*MAX_DIM*MAX_DIM, 64);
	double* hess_vt = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* hess_vt_origin = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* diff_hess_vt = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* vt = Util::get_randn(MAX_DIM, 0, 10);
	int* idx = Util::get_rand_choice(sample_size, 0, N);
	Model* model = new LSvm(X, Y, N, 1e-4);
	model->init_model();
	model->get_hessian(idx, sample_size, hess);
	model->get_hess_vt(idx, sample_size, vt, hess_vt_origin);
	double hess_norm = squared_norm(MAX_DIM*MAX_DIM, hess);
	double vt_norm = squared_norm(MAX_DIM, vt);
	double sum = cblas_dasum(MAX_DIM*MAX_DIM, hess, 1);
	hess_vt_origin_norm = squared_norm(MAX_DIM, hess_vt_origin);
	printf("hess sumis:%.15f", sum);
	printf("hessian norm is%.15f\n", hess_norm);
	printf("vt norm is%.15f\n", vt_norm);
	gemv(RowMajor, NoTrans, MAX_DIM, MAX_DIM, 1.0, hess, MAX_DIM, vt, 1, 0, hess_vt, 1);
	vdSub(MAX_DIM, hess_vt, hess_vt_origin, diff_hess_vt);
	hess_vt_norm = squared_norm(MAX_DIM, hess_vt);
	hess_vt_origin_norm = squared_norm(MAX_DIM, hess_vt_origin);
	error = squared_norm(MAX_DIM, diff_hess_vt);
	relative_error = error / mi(hess_vt_norm, hess_vt_origin_norm);

	printf("hessian vt by hess_vt norm is%.15f\n", hess_vt_origin_norm);
	printf("hessian vt by hess@vt norm is%.15f\n", hess_vt_norm);
	printf("error hessian vt  norm is%.15f\n", error);
	printf("relative error hessian vt  norm is%.15f\n", relative_error);

	mkl_free(hess);
	mkl_free(hess_vt);
	mkl_free(hess_vt_origin);
	mkl_free(diff_hess_vt);
	mkl_free(vt);
	mkl_free(idx);
	// delete[] model;
}

void Test::test_lant_hess_dot_approx(double * X, double * Y, int N, int sample_size, int rank)
{
	printf("-----------------------------------\n");
	printf("test_lant_hess_dot_approx\n");
	double hess_norm = 0, approx = 0, exact = 0, error = 0, relative_error = 0;
	double* grad = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* hess = (double*)mkl_malloc(sizeof(double)*MAX_DIM*MAX_DIM, 64);
	double* hess_matrix_approx = (double*)mkl_malloc(sizeof(double)*MAX_DIM*rank, 64);
	double* hess_matrix_exact = (double*)mkl_malloc(sizeof(double)*MAX_DIM*rank, 64);
	double* hess_matrix_diff = (double*)mkl_malloc(sizeof(double)*MAX_DIM*rank, 64);
	double* matrix = Util::get_randn(MAX_DIM*rank, 0, 0.1);
	int* idx = Util::get_rand_choice(sample_size, 0, N);
	Model* model = new Logistic(X, Y, N, 1e-4);
	model->init_model();
	model->get_hessian(idx, sample_size, hess);
	model->get_gradient(idx, sample_size, grad);
	gemm(RowMajor, NoTrans, NoTrans, rank, MAX_DIM, MAX_DIM, 1.0, matrix, MAX_DIM,hess, MAX_DIM, 0, hess_matrix_exact, MAX_DIM);
	int stage_a = 3, stage_b = 3;
	Lant* lant = new Lant(model, 1e-5, rank, sample_size, stage_a, stage_b, 10,  1, true);
	lant->get_hess_dot_matrix_approx(matrix, hess_matrix_approx, idx, sample_size, grad, rank);

	vdSub(MAX_DIM*rank, hess_matrix_approx, hess_matrix_exact, hess_matrix_diff);
	hess_norm = squared_norm(MAX_DIM*MAX_DIM, hess);
	approx = squared_norm(MAX_DIM*rank, hess_matrix_approx);
	exact = squared_norm(MAX_DIM*rank, hess_matrix_exact);
	error = squared_norm(MAX_DIM*rank, hess_matrix_diff);
	relative_error = error / mi(approx, exact);

	printf("hess norm is %.15f\n", hess_norm);
	printf("exact norm is %.15f\n", exact);
	printf("approx norm is %.15f\n", approx);
	printf("error norm is %.15f\n", error);
	printf("relative_error norm is %.15f\n", relative_error);


	mkl_free(grad);
	mkl_free(matrix);
	mkl_free(hess_matrix_approx);
	mkl_free(hess_matrix_diff);
	mkl_free(hess_matrix_exact);
	mkl_free(hess);
	mkl_free(idx);
	delete[] model;
	delete[] lant;
}

void Test::test_logistic_get_gradient_multi_ndim(double * X, double * Y, int N, int rank)
{
	printf("---------------------test_logistic_get_gradient_multi_ndim-----------------------\n\n");
	Model* model = new LSvm(X, Y, N, 1e-4);
	double* grad = (double*)mkl_malloc(sizeof(double)*MAX_DIM*rank, 64);
	double* w = Util::get_randn(MAX_DIM*rank, 0, 1e-4);
	double* new_w = (double*)mkl_malloc(sizeof(double)*MAX_DIM*rank, 64);
	double* delta_w = Util::get_randn(MAX_DIM*rank, 0, 1e-4);
	double* loss = (double*)mkl_malloc(sizeof(double)*rank, 64);
	double* new_loss = (double*)mkl_malloc(sizeof(double)*rank, 64);
	double* diff_loss = (double*)mkl_malloc(sizeof(double)*rank, 64);
	double* diff_loss_approx = (double*)mkl_malloc(sizeof(double)*rank, 64);
	double* error = (double*)mkl_malloc(sizeof(double)*rank, 64);
	double* relative_error = (double*)mkl_malloc(sizeof(double)*rank, 64);
	int* idx = (int*)mkl_malloc(sizeof(int)*N, 64);
	for(int i =0; i < N; ++i) idx[i] = i;
	vdAdd(MAX_DIM*rank, w, delta_w, new_w);
	model->get_gradient(idx, N, grad, w, rank);
	for (int i = 0; i < rank; ++i) {
		loss[i] = model->get_loss(w + MAX_DIM * i);
		new_loss[i] = model->get_loss(new_w + MAX_DIM * i);
		diff_loss[i] = new_loss[i]- loss[i];
		diff_loss_approx[i] = dot(MAX_DIM, grad + MAX_DIM * i, 1, delta_w + MAX_DIM * i, 1);
		error[i] = diff_loss[i] - diff_loss_approx[i];
		relative_error[i] = error[i] / 1e-4;
		printf("diff loss:%.10f\n", diff_loss[i]);
		printf("diff loss approx :%.10f\n", diff_loss_approx[i]);
		printf("error:%.10f\n", error[i]);
		printf("relative_error:%.10f\n", relative_error[i]);
		printf("----------------------------------\n");
	}


	
	mkl_free(idx);
	mkl_free(grad);
	mkl_free(w);
	mkl_free(new_w);
	mkl_free(delta_w);
	mkl_free(loss);
	mkl_free(new_loss);
	mkl_free(diff_loss);
	mkl_free(diff_loss_approx);
	mkl_free(error);
	mkl_free(relative_error);
	delete[] model;
}

void Test::test_run_stage_a_4_3(double * X, double * Y, int N, int rank)
{
	printf("-----------------------------------\n");
	printf("test_run_stage_a_4_3\n");
	int sample_size = N;
	rank = ma(rank, 40);
	int more_rank = 20, keep_rank = rank - more_rank;
	double* grad = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* hess = (double*)mkl_malloc(sizeof(double)*MAX_DIM*MAX_DIM, 64);
	double* rand_matrix = Util::get_randn(MAX_DIM*rank, 0, 0.1);
	double* y_matrix = (double*)mkl_malloc(sizeof(double)*MAX_DIM*rank, 64);
	double* y_matrix_exact = (double*)mkl_malloc(sizeof(double)*MAX_DIM*rank, 64);
	double* y_matrix_diff = (double*)mkl_malloc(sizeof(double)*MAX_DIM*rank, 64);
	double* q_matrix = (double*)mkl_malloc(sizeof(double)*MAX_DIM*rank, 64);
	double* q_matrix_hess = (double*)mkl_malloc(sizeof(double)*MAX_DIM*MAX_DIM, 64);
	double* q_matrix_dot = (double*)mkl_malloc(sizeof(double)*(MAX_DIM-rank)*keep_rank, 64);
	double* s = (double*)MKL_malloc(sizeof(double)*MAX_DIM, 64);
	int* idx = Util::get_rand_choice(sample_size, 0, N);
	double* q_matrix_exact = q_matrix_hess + MAX_DIM * (MAX_DIM - keep_rank);
	double* q_matrix_orth = q_matrix_hess;
	double* q_matrix_keep = q_matrix + MAX_DIM * more_rank;
	Model* model = new Logistic(X, Y, N, 1e-4);
	model->init_model();
	model->get_hessian(idx, sample_size, hess);
	model->get_gradient(idx, sample_size, grad);
	int stage_a = 3, stage_b = 3;
	Lant* lant = new Lant(model, 1e-5, keep_rank, sample_size,stage_a, stage_b, more_rank, 1, false);

	// 对hess进行SVD
	mem_copy(MAX_DIM*MAX_DIM, hess, 1, q_matrix_hess, 1);
	LAPACKE_dsyevd(CblasRowMajor, 'V', 'U', MAX_DIM, q_matrix_hess, MAX_DIM, s);
	transpose('R', 'T', MAX_DIM, MAX_DIM, 1.0, q_matrix_hess, MAX_DIM, MAX_DIM);
	// s 是升序， 且q_matrix是列向量方式， 所以需要转置, q_matrix+MAX_DIM*(MAX_DIM-rank)是最大rank个特征向量

	lant->run_stage_a_4_3(rand_matrix, idx, sample_size, grad, q_matrix, y_matrix);

	//check y_matrix
	gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rank, MAX_DIM, MAX_DIM, 1.0, rand_matrix, MAX_DIM,hess, MAX_DIM,
		0, y_matrix_exact, MAX_DIM);

	vdSub(MAX_DIM*rank, y_matrix_exact, y_matrix, y_matrix_diff);
	double y_matrix_norm = 0, y_matrix_exact_norm = 0, y_matrix_diff_norm = 0;
	y_matrix_norm = squared_norm(MAX_DIM*rank, y_matrix);
	y_matrix_exact_norm = squared_norm(MAX_DIM*rank, y_matrix_exact);
	y_matrix_diff_norm = squared_norm(MAX_DIM*rank, y_matrix_diff);
	printf("y_matrix norm is:      %.15f\n", y_matrix_norm);
	printf("y_matrix_exact norm is:%.15f\n", y_matrix_exact_norm);
	printf("y_matrix_diff norm is: %.15f\n", y_matrix_diff_norm);


	//check q_matrix@q_matrix_orth
	gemm(CblasRowMajor, CblasNoTrans, CblasTrans, MAX_DIM - rank, keep_rank, MAX_DIM, 1.0, q_matrix_orth, MAX_DIM, 
		q_matrix_keep, MAX_DIM, 0, q_matrix_dot, keep_rank);
	double q_matrix_keep_norm = 0, q_matrix_orth_norm = 0, q_matrix_dot_norm = 0;
	q_matrix_keep_norm = squared_norm(MAX_DIM*keep_rank, q_matrix_keep);
	q_matrix_orth_norm = squared_norm(MAX_DIM*(MAX_DIM - rank), q_matrix_orth);
	q_matrix_dot_norm = squared_norm(keep_rank*(MAX_DIM - rank), q_matrix_dot);
	printf("q_matrix_keep norm is: %.15f\n", q_matrix_keep_norm);
	printf("q_matrix_orth norm is: %.15f\n", q_matrix_orth_norm);
	printf("q_matrix_dot  norm is: %.4f\n", q_matrix_dot);




	mkl_free(grad);
	mkl_free(hess);
	mkl_free(rand_matrix);
	mkl_free(y_matrix);
	mkl_free(y_matrix_diff);
	mkl_free(y_matrix_exact);
	mkl_free(q_matrix_hess);
	mkl_free(q_matrix);
	mkl_free(q_matrix_dot);
	mkl_free(s);
	mkl_free(idx);
	delete[] model;
	delete[] lant;
}

void Test::test_run_stage_a_5_3(double * X, double * Y, int N, int rank)
{
	printf("-----------------------------------\n");
	printf("test_run_stage_b_5_3\n");
	int sample_size = N;
	rank = ma(rank, 40);
	int more_rank = 20, keep_rank = rank - more_rank;
	double* grad = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* hess = (double*)mkl_malloc(sizeof(double)*MAX_DIM*MAX_DIM, 64);
	double* hess_ksvd = (double*)mkl_malloc(sizeof(double)*MAX_DIM*MAX_DIM, 64);
	double* hess_lant = (double*)mkl_malloc(sizeof(double)*MAX_DIM*MAX_DIM, 64);
	double* hess_diff = (double*)mkl_malloc(sizeof(double)*MAX_DIM*MAX_DIM, 64);
	double* q_matrix = (double*)mkl_malloc(sizeof(double)*MAX_DIM*MAX_DIM, 64);
	double* u = (double*)mkl_malloc(sizeof(double)*MAX_DIM*rank, 64);
	double* v = (double*)mkl_malloc(sizeof(double)*MAX_DIM*rank, 64);
	double* s = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	int* idx = Util::get_rand_choice(sample_size, 0, N);
	Model* model = new Logistic(X, Y, N, 1e-4);
	model->init_model();
	model->get_hessian(idx, sample_size, hess);
	model->get_gradient(idx, sample_size, grad);

	// 对hess进行SVD
	mem_copy(MAX_DIM*MAX_DIM, hess, 1, q_matrix, 1);
	LAPACKE_dsyevd(CblasRowMajor, 'V', 'U', MAX_DIM, q_matrix, MAX_DIM, s);
	transpose('R', 'T', MAX_DIM, MAX_DIM, 1.0, q_matrix, MAX_DIM, MAX_DIM);
	// s 是升序， 且q_matrix是列向量方式， 所以需要转置, q_matrix+MAX_DIM*(MAX_DIM-rank)是最大rank个特征向量

	//求hess_ksvd
	mem_copy(MAX_DIM*keep_rank, q_matrix + MAX_DIM * (MAX_DIM - keep_rank), 1, u, 1);
	mem_copy(MAX_DIM*keep_rank, q_matrix + MAX_DIM * (MAX_DIM - keep_rank), 1, v, 1);
	for (int i = 0; i < keep_rank; ++i) {
		scal(MAX_DIM, s[i+MAX_DIM- keep_rank], u + MAX_DIM * i, 1);
	}
	gemm(RowMajor, Trans, NoTrans, MAX_DIM, MAX_DIM, keep_rank, 1.0, u , MAX_DIM, v, MAX_DIM,
		0, hess_ksvd, MAX_DIM);

	int stage_a = 3, stage_b = 3;
	Lant* lant = new Lant(model, 1e-5, keep_rank, sample_size, stage_a, stage_b, more_rank,  1, true);
	//输入最大的rank对应的特征向量当作正交基
	lant->run_stage_b_5_3(q_matrix + MAX_DIM * (MAX_DIM - rank), idx, sample_size, grad, u, s);
	
	//求hess_lant
	mem_copy(MAX_DIM*rank, u, 1, v, 1);
	for (int i = 0; i < rank; ++i) {
		scal(MAX_DIM, s[i], u + MAX_DIM * i, 1);
	}
	Util::print_matrix("s", s, 1, rank, rank);
	gemm(RowMajor, Trans, NoTrans, MAX_DIM, MAX_DIM, keep_rank, 1.0, u + MAX_DIM * more_rank, MAX_DIM, v + MAX_DIM * more_rank, MAX_DIM,
		0, hess_lant, MAX_DIM);

	vdSub(MAX_DIM*MAX_DIM, hess_lant, hess_ksvd, hess_diff);
	double hess_norm = 0, hess_ksvd_norm = 0, hess_lant_norm = 0, hess_diff_norm = 0;
	hess_norm = squared_norm(MAX_DIM * MAX_DIM, hess);
	hess_ksvd_norm = squared_norm(MAX_DIM * MAX_DIM, hess_ksvd);
	hess_lant_norm = squared_norm(MAX_DIM * MAX_DIM, hess_lant);
	hess_diff_norm = squared_norm(MAX_DIM * MAX_DIM, hess_diff);

	printf("hess      norm is %.15f\n", hess_norm);
	printf("hess ksvd norm is %.15f\n", hess_ksvd_norm);
	printf("hess lant norm is %.15f\n", hess_lant_norm);
	printf("hess diff norm is %.15f\n", hess_diff_norm);
	printf("relative error is %.15f\n", hess_diff_norm / ma(1e-16, mi(hess_ksvd_norm, hess_lant_norm)));

	mkl_free(grad);
	mkl_free(hess);
	mkl_free(hess_ksvd);
	mkl_free(hess_lant);
	mkl_free(hess_diff);
	mkl_free(u);
	mkl_free(s);
	mkl_free(v);
	mkl_free(q_matrix);
	mkl_free(idx);
	delete[] model;
	delete[] lant;
}

void Test::test_ksvd()
{
	const int N = 5, NSELECT = 3, LDA = 5, LDZ = NSELECT;
	MKL_INT n = N, il, iu, m, lda = LDA, ldz = LDZ, info;
	double abstol, vl = 0.0, vu = 0.0;
	/* Local arrays */
	MKL_INT isuppz[2 * N];
	double w[N], z[LDZ*N];
	double a[LDA*N] = {
		0.67, -0.20, 0.19, -1.06, 0.46,
		0.00,  3.82, -0.13,  1.06, -0.48,
		0.00,  0.00, 3.27,  0.11, 1.10,
		0.00,  0.00, 0.00,  5.86, -0.98,
		0.00,  0.00, 0.00,  0.00, 3.54
	};
	/* Executable statements */
	printf("LAPACKE_dsyevr (row-major, high-level) Example Program Results\n");
	/* Negative abstol means using the default value */
	abstol = -1.0;
	/* Set il, iu to compute NSELECT smallest eigenvalues */
	il = 3;
	iu = 2+NSELECT;
	/* Solve eigenproblem */
	info = LAPACKE_dsyevr(LAPACK_ROW_MAJOR, 'V', 'I', 'U', n, a, lda,
		vl, vu, il, iu, abstol, &m, w, z, ldz, isuppz);
	/* Check for convergence */
	if (info > 0) {
		printf("The algorithm failed to compute eigenvalues.\n");
		exit(1);
	}
	/* Print the number of eigenvalues found */
	printf("\n The total number of eigenvalues found:%2i\n", m);
	/* Print eigenvalues */
	Util::print_matrix("Selected eigenvalues",w, 1, m, 1);
	/* Print eigenvectors */
	Util::print_matrix("Selected eigenvectors (stored columnwise)",z, n, m,ldz);
}

void Test::test_random_ksvd()
{
	const int N = 3;
	//double* a = (double*)mkl_malloc(sizeof(double)*N*N, 64);
	//mem_zero(N*N, a);
	//for (int i = 0; i < N*N; ++i) a[i] = i;
	double a[9] = { 1,4,9,4, 25, 36, 9, 36, 81 };
	Util::print_matrix("a", a, N, N, N);
	double nrm = Util::spectral_norm(a, N);
	printf("the spectral_norm of a is%.10f\n", nrm);


	//only for once
	ifstream file;
	file.open("..//..//LR//resource//hess_diff1.500000");
	const int d = 785;
	double* data = (double*)mkl_malloc(sizeof(double) * 785*785, 64);
	char ch;
	for (int i = 0; i < d; ++i) {
		for (int j = 0; j < d; ++j) {
			file >> data[i*d + j];
			if (j!=d-1) {
				file >> ch;
			}
		}
	}
	Util::print_matrix("data", data, 10, 10, d);
	double nrm1 = Util::norm(data, d);
	double nrm2 = Util::spectral_norm(data, d);
	printf("random norm is:%.16f\n", nrm1);
	printf("spectral_norm norm is:%.16f\n", nrm2);
	mkl_free(data);
}

void Test::test_symm_inv(int N)
{
	double* rand_matrix = Util::get_randn(N*N, 0, 1);
	double* A = (double*)mkl_malloc(sizeof(double)*N*N, 64);
	double* A_inv = (double*)mkl_malloc(sizeof(double)*N*N, 64);
	double* out = (double*)mkl_malloc(sizeof(double)*N*N, 64);
	double* tmp = (double*)mkl_malloc(sizeof(double)*N, 64);
	gemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, rand_matrix, N, rand_matrix, N, 0, A, N);
	Util::print_matrix("A", A, N, N, N);
	mem_copy(N*N, A, 1, A_inv, 1);
	LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', N, A_inv, N);
	int ret = LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'U', N, A_inv, N);
	printf("ret:%d\n", ret);
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < i; ++j) {
			A_inv[i*N + j] = A_inv[j*N + i];
		}
	}

	gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, A_inv, N, 0, out, N);
	//cblas_dsymm(CblasRowMajor, CblasRight, CblasUpper, N, N, 1, A, N, A_inv, N, 0, out, N);
	double a_norm = 0, a_inv_norm, out_norm = 0;
	a_norm = squared_norm(N*N, A);
	a_inv_norm = squared_norm(N*N, A_inv);
	out_norm = squared_norm(N*N, out);
	printf("A norm is %.10f\n", a_norm);
	printf("A_inv norm is %.10f\n", a_inv_norm);
	printf("A@A_inv norm is %.10f\n", out_norm);
	Util::print_matrix("A_inv", A_inv, N, N, N);
	Util::print_matrix("A@A_inv", out, N, N, N);


	mkl_free(tmp);
	mkl_free(rand_matrix);
	mkl_free(out);
	mkl_free(A_inv);
	mkl_free(A);
}

void Test::test_inv(int N)
{
	double* rand_matrix = Util::get_randn(N*N, 0, 1);
	double* A = (double*)mkl_malloc(sizeof(double)*N*N, 64);
	double* A_inv = (double*)mkl_malloc(sizeof(double)*N*N, 64);
	double* out = (double*)mkl_malloc(sizeof(double)*N*N, 64);
	int* tmp = (int*)mkl_malloc(sizeof(int)*N, 64);
	gemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, rand_matrix, N, rand_matrix, N, 0, A, N);
	Util::print_matrix("A", A, N, N, N);
	mem_copy(N*N, A, 1, A_inv, 1);
	
	LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, A_inv, N, tmp);
	LAPACKE_dgetri(LAPACK_ROW_MAJOR, N, A_inv, N, tmp);
	gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, A_inv, N, 0, out, N);
	//cblas_dsymm(CblasRowMajor, CblasRight, CblasUpper, N, N, 1, A, N, A_inv, N, 0, out, N);
	double a_norm = 0, a_inv_norm, out_norm = 0;
	a_norm = squared_norm(N*N, A);
	a_inv_norm = squared_norm(N*N, A_inv);
	out_norm = squared_norm(N*N, out);
	printf("A norm is %.10f\n", a_norm);
	printf("A_inv norm is %.10f\n", a_inv_norm);
	printf("A@A_inv norm is %.10f\n", out_norm);
	Util::print_matrix("A_inv", A_inv, N, N, N);
	Util::print_matrix("A@A_inv", out, N, N, N);


	mkl_free(tmp);
	mkl_free(rand_matrix);
	mkl_free(out);
	mkl_free(A_inv);
	mkl_free(A);
}
void Test::test_model(){
	double *X = nullptr, *Y = nullptr;
	int N = Util::loadMnist49(&X, &Y);
	test_get_hessian_mm();
	test_get_hess_vt_mm();
	test_get_gradient_mm();
	test_logistic_get_gradient(X, Y, N);
	test_logistic_hess_vt(X, Y, N);
	test_logistic_hessian(X, Y, N, 1000);
	test_logistic_get_gradient_multi_ndim(X, Y, N, 30);
	mkl_free(X);
	mkl_free(Y);
}
void Test::test_openmpi(int m, int n, int thread_num)
{
	double* data = (double*)malloc(sizeof(double)*n);
	double* sum1 = (double*)malloc(sizeof(double)*n);
	double* sum2 = (double*)malloc(sizeof(double)*n*thread_num);
	for (int i = 0; i < n; ++i) {
		data[i] = 0.01*(i % 100);
	}
	memset(sum1, 0, sizeof(double)*n);
	memset(sum2, 0, sizeof(double)*n*thread_num);
	auto st1 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < m; ++i) {
		double* data_i = data;
		for (int j = 0; j < n; ++j) {
			sum1[j] += data_i[j];
		}
	}
	auto ed1 = std::chrono::high_resolution_clock::now();
	auto st2 = std::chrono::high_resolution_clock::now();
#pragma	omp parallel for num_threads(thread_num)
	for (int i = 0; i < m; ++i) {
		double* data_i = data;
		double* sum_i = sum2 + (omp_get_thread_num()*n);
		for (int j = 0; j < n; ++j) {
			sum_i[j] += data_i[j];
		}
	}
	auto ed2 = std::chrono::high_resolution_clock::now();
	printf("sum1 cost time:%lld\n", (ed1 - st1).count());
	printf("sum2 cost time:%lld\n", (ed2 - st2).count());
	for (int i = 1; i < thread_num; ++i) {
		double* sum_i = sum2 + i * n;
		for (int j = 0; j < n; ++j) {
			sum2[j] += sum_i[j];
		}
	}
	double error = 0, sum = 0;
	for (int i = 0; i < n; ++i) {
		sum += sum1[i];
		error += abs(sum1[i] - sum2[i]);
	}
	printf("error is:%.10f\n", error);
	printf("sum is:%.10f\n", sum);
	free(sum1);
	free(sum2);
	free(data);
}
void Test::test_openmpi_mkl(int m, int n, int thread_num)
{
	double* data = Util::get_randn(m*n, 0, 1);
	double* sum1 = (double*)mkl_malloc(sizeof(double)*n, 64);
	double* sum2 = (double*)mkl_malloc(sizeof(double)*n*thread_num, 64);
	mem_zero(n, sum1);
	mem_zero(n*thread_num, sum2);
	auto st1 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < m; ++i) {
		axpy(n, 1.0, data + i * n, 1, sum1, 1);
	}
	auto ed1 = std::chrono::high_resolution_clock::now();
	auto st2 = std::chrono::high_resolution_clock::now();
	#pragma	omp parallel for num_threads(thread_num)
	for (int i = 0; i < m; ++i) {
		double* sum_i = sum2 + (omp_get_thread_num()*n);
		axpy(n, 1.0, data + i * n, 1, sum_i, 1);
	}
	auto ed2 = std::chrono::high_resolution_clock::now();
	for (int i = 1; i < thread_num; ++i) {
		double* sum_i = sum2 + i * n;
		axpy(n, 1.0, sum_i, 1, sum2, 1);
	}
	printf("sum1 cost time:%.5f s\n", (chrono::duration_cast<chrono::duration<double>>(ed1 - st1)).count());
	printf("sum2 cost time:%.5f s\n", (chrono::duration_cast<chrono::duration<double>>(ed2 - st2)).count());
	double error = 0, sum = 0;
	for (int i = 0; i < n; ++i) {
		sum += sum1[i];
		error += abs(sum1[i] - sum2[i]);
	}
	printf("error is:%.10f\n", error);
	printf("sum is:%.10f\n", sum);

	mkl_free(sum1);
	mkl_free(sum2);
	mkl_free(data);
}

void Test::test_logistic_hess_vt(double* X, double* Y, int N)
{
	printf("---------------------test_logistic_hess_vt-----------------------\n");
	double error = 0, relative_error = 0;
	Model* model = new LSvm(X, Y, N, 1e-4);
	double* grad = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* new_grad = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* delta_grad = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* hess_vt = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* diff = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
	double* delta_w = Util::get_randn(MAX_DIM, 0, 1e-5);
	double grad_norm = 0, new_grad_norm = 0, hess_vt_norm = 0, diff_norm = 0, delta_grad_norm = 0, delta_w_norm = 0;
	int* idx = (int*)mkl_malloc(sizeof(int)*(N), 64);
	for (int i = 0; i < N; ++i) idx[i] = i;

	model->init_model();
	model->get_gradient(idx, N, grad);
	model->update_model_by_add(delta_w);
	model->get_gradient(idx, N, new_grad);
	model->get_hess_vt(idx, N, delta_w, hess_vt);
	vdSub(MAX_DIM, new_grad, grad, delta_grad);
	vdSub(MAX_DIM, hess_vt, delta_grad, diff);
	delta_w_norm = squared_norm(MAX_DIM, delta_w);
	grad_norm = squared_norm(MAX_DIM, grad);
	new_grad_norm = squared_norm(MAX_DIM, new_grad);
	delta_grad_norm = squared_norm(MAX_DIM, delta_grad);
	hess_vt_norm = squared_norm(MAX_DIM, hess_vt);
	diff_norm = squared_norm(MAX_DIM, diff);
	error = diff_norm, relative_error = error / mi(hess_vt_norm, delta_grad_norm);
	printf("delta_w norm2 is:%.25f\n", delta_w_norm);
	printf("grad norm2 is   :%.25f\n", grad_norm);
	printf("new_grad norm2  :%.25f\n", new_grad_norm);
	printf("delta_grad norm2:%.25f\n", delta_grad_norm);
	printf("hess_vt norm2 is:%.25f\n", hess_vt_norm);
	printf("error           :%.25f\n", error);
	printf("relative_error  :%.25f\n", relative_error);
	mkl_free(delta_w);
	mkl_free(grad);
	mkl_free(delta_grad);
	mkl_free(new_grad);
	mkl_free(hess_vt);
	mkl_free(diff);
	mkl_free(idx);
}

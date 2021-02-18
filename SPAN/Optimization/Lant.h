#pragma once
#include "Optimizer.h"

class Lant :
	public Optimizer
{
public:
	Lant(Model* _model, double scale_w, int keep_rank, int sample_size, int stage_a, int stage_b, int more_rank, 
		double _step_size, bool use_hess_vt = false, int power_iter=3) : Optimizer(_model, "lant", _step_size),
		scale_w(scale_w),keep_rank(keep_rank), sample_size(sample_size), stage_a(stage_a), 
		stage_b(stage_b),more_rank(more_rank), use_hess_vt(use_hess_vt), power_iter(power_iter){
		one_dim = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
		for (int i = 0; i < MAX_DIM; ++i) one_dim[i] = 1;

		params.set_param("scale_w", scale_w);
		params.set_param("keep_rank", keep_rank);
		params.set_param("sample_size", sample_size);
		params.set_param("stage_a", stage_a);
		params.set_param("stage_b", stage_b);
		params.set_param("more_rank", more_rank);
		params.set_param("use_hess_vt", use_hess_vt);
		params.set_param("power_iter", power_iter);
	};
	Lant(Model* _model, const Params& _params):Optimizer(_model, "lant", _params) {
		one_dim = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
		for (int i = 0; i < MAX_DIM; ++i) one_dim[i] = 1;


		use_hess_vt = (bool)params.get_param("use_hess_vt");
		keep_rank = (int)params.get_param("keep_rank");
		more_rank = (int)params.get_param("more_rank");
		sample_size = (int)params.get_param("sample_size");
		scale_w = params.get_param("scale_w");
		stage_a = params.get_param("stage_a");
		stage_b = params.get_param("stage_b");
		power_iter = params.get_param("power_iter");
	};
	virtual Records run(int iter_num, Records records, double * weights = nullptr);
	void run_stage_a_4_3(const double* rand_matrix,const  int* idx, int n_samples, const double* grad, double* q_matrix, double* y_matrix, const double* weights = nullptr, int _power_iter=1);
	void run_stage_b_5_3(const double* q_matrix, int* idx, int n_samples, const double* grad, double* u, double* s, const double* weights = nullptr);
	void run_stage_ab(const double* rand_matrix, int * idx, int n_samples, double* u, double* s, double* weight=nullptr);
	void get_hess_dot_matrix_approx(const double* matrix, double* out, const int idx[], int n_samples,const  double* grad, int ndim, const double* weights=nullptr, int _power_iter=0);
	void get_hess_approx(double* u, double* s, int* idx=nullptr, double* weight = nullptr);
	void get_hess_inv_vt_by_bfgs(double* u, double* s, double * s_array, double * y_array, double* vt, double* res, int now_store_num);
	void get_hess_inv_vt_by_usv(double* u, double* s, double* vt, double* hess_inv_vt);
	~Lant();

	double* one_dim;
	bool use_hess_vt;
	int keep_rank;
	int more_rank;
	int sample_size;
	double scale_w;
	int stage_a;
	int stage_b;
	int power_iter;
};


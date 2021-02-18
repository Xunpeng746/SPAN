#pragma once
#include "Optimizer.h"
class Sbbfgs :
	public Optimizer
{
public:
	Sbbfgs(Model* model,  double step_size, int rank, int b, int bh, int m, int store_num) :
		Optimizer(model, "Sbbfgs", step_size), rank(rank), b(b), bh(bh), m(m), store_num(store_num){
		params.set_param("rank", rank);
		params.set_param("b", b);
		params.set_param("bh", bh);
		params.set_param("m", m);
		params.set_param("store_num", store_num);
	};
	Sbbfgs(Model* model, const Params& _params) :Optimizer(model, "sbbfgs", _params) {
		rank = (int)params.get_param("rank");
		b = (int)params.get_param("b");
		bh = (int)params.get_param("bh");
		m = (int)params.get_param("m");
		store_num = (int)params.get_param("store_num");
	};
	~Sbbfgs();
	virtual Records run(int iter_num, Records records, double * weights = nullptr);
	void get_block_bfgs_update(double * d_array, double * y_array, double * delta_array, double* vt, double* res, int now_store_num);
	void get_hess_inv_approx(double * d_array, double * y_array, double * delta_array,  double* res, int now_store_num);
	int rank;
	int b;
	int bh;
	int m;
	int store_num;
};


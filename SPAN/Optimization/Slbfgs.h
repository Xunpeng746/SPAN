#pragma once
#include "Optimizer.h"
class Slbfgs :
	public Optimizer
{
public:
	virtual Records run(int iter_num, Records records, double * weights = nullptr);
	Slbfgs(Model* model, double step_size, int store_num, int update_period, int b,
		int bh, int m) :Optimizer(model, "Slfbgs", step_size), store_num(store_num), update_period(update_period),
		b(b), bh(bh), m(m) {
		ones = (double*)mkl_malloc(sizeof(double)*ma(update_period, store_num), 64);

		params.set_param("update_period", update_period);
		params.set_param("b", b);
		params.set_param("bh", bh);
		params.set_param("m", m);
		params.set_param("store_num", store_num);
	}
	Slbfgs(Model* model, const Params& _params) :Optimizer(model, "Slbfgs", _params) {
		ones = (double*)mkl_malloc(sizeof(double)*ma(update_period, store_num), 64);

		update_period = (int)params.get_param("update_period");
		b = (int)params.get_param("b");
		bh = (int)params.get_param("bh");
		m = (int)params.get_param("m");
		store_num = (int)params.get_param("store_num");
	}

	void get_hess_inv_vt_approx(double * s_array, double * y_array,  double* vt, double* res, int now_store_num);
	void get_hess_inv_approx(double * s_array, double * y_array,  double* res, int now_store_num);
	~Slbfgs();



	int store_num;
	int update_period;
	int b;
	int bh;
	int m;
	double* ones;
};


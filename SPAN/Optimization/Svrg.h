#pragma once
#include "Optimizer.h"
class Svrg :
	public Optimizer
{
public:
	Svrg(Model* model, std::string optimizer_name, double step_size, int m) :
		Optimizer(model, optimizer_name, step_size), m(m) {
		params.set_param("m", m);
	};
	Svrg(Model* model, const Params& _params) :Optimizer(model, "Svrg", _params) {
		m = params.get_param("m");
	};
	~Svrg();
	virtual Records run(int iter_num, Records records, double * weights = nullptr) ;

	int m;
};


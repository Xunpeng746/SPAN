#pragma once
#include "Optimizer.h"
class GD :
	public Optimizer
{
public:

	GD(Model* _model, double _step_size):Optimizer(_model, "GD", _step_size){

	};
	GD(Model* _model, const Params& _params) :Optimizer(_model, "GD", _params) {

	};
	virtual Records run(int iter_num, Records records, double * weights = nullptr);
	~GD();
};


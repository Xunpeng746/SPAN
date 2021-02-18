#pragma once
#include "Optimizer.h"
class Lissa :
	public Optimizer
{
public:
	~Lissa();
	Lissa(Model* _model, int _t1, int _s1, int _s2, double _step_size) :
		Optimizer(_model, "Lissa", _step_size), t1(_t1), s1(_s1), s2(_s2) {
		params.set_param("t1", t1);
		params.set_param("s1", s1);
		params.set_param("s2", s2);
	} 
	Lissa(Model* _model, const Params& _params):Optimizer(_model, "Lissa", _params){
		t1 = (int)params.get_param("t1");
		s1 = (int)params.get_param("s1");
		s2 = (int)params.get_param("s2");
	}
	Records run(int iter_num, Records records, double * weights = nullptr);
	double get_hess_value(int idx, double*& data_x);
	int t1;
	int s1;
	int s2;
};


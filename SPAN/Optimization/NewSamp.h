#pragma once
#include "Optimizer.h"
#include "MemFactory.hpp"
extern size_t THREAD_NUM_MKL;
class NewSamp :
	public Optimizer
{
public:
	NewSamp(Model* model, double step_size, int rank, int sample_size) :
		Optimizer(model, "NewSamp", step_size), rank(rank), sample_size(sample_size) {
		params.set_param("rank", rank);
		params.set_param("sample_size", sample_size);
	};
	NewSamp(Model* model, const Params& _params):
		Optimizer(model, "NewSamp", _params){
		rank = (int)params.get_param("rank");
		sample_size = (int)params.get_param("sample_size");
	};
	~NewSamp();
	virtual Records run(int iter_num, Records records, double * weights = nullptr);
	int rank;
	int sample_size;
};

 
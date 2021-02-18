#include "Optimizer.h"


Optimizer::~Optimizer()
{
}

double Optimizer::log_iter_end(chrono::time_point<chrono::high_resolution_clock> & start, double data_pass, Records& records)
{
	auto now = std::chrono::high_resolution_clock::now();
	double loss = model->get_loss();
	Record record( (double)(chrono::duration_cast<chrono::duration<double>>(now - start)).count(), data_pass*1.0, loss );
	if(records.record_list.size()>0){
		int last = records.record_list.size()-1;
		record.hess_error = records.record_list[last].hess_error;
	}
	records.push_back(record);
	
	start += std::chrono::high_resolution_clock::now() - now;
	record.print();
	return loss;
}

void Optimizer::step(double step_size, double * data)
{
	model->step(step_size, data);
}

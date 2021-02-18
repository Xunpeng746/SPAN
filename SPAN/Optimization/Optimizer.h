#pragma once
#include "config.h"
#include "Model.h"
#include <time.h>
#include <string>
#include <vector>
#include "Param.hpp"
#include "Records.hpp"
extern bool PRINT_HESS;
class Optimizer
{
public:
	Optimizer(Model* _model, std::string _optimizer_name, const Params& _params) :model(_model), optimizer_name(_optimizer_name),params(_params) {
		lambda = model->m_lambda[0];
		N = model->N;
		step_size = params.get_param("step_size");
		params.set_param("N", N);
		params.set_param("lambda", lambda);
		compare_hess = false;
	}
	Params get_params() {
		return params;
	}
	Optimizer(Model* _model, std::string _optimizer_name, double _step_size) :
		model(_model), optimizer_name(_optimizer_name), step_size(_step_size) {
		lambda = model->m_lambda[0];
		N = model->N;
		params.set_param("step_size", step_size);
		params.set_param("N", N);
		params.set_param("lambda", lambda);
		compare_hess = true;
	}

	void log_params() {
		params.log_params();
	}
	
	virtual Records run(int iter_num, Records records, double * weights = nullptr)=0;
	virtual ~Optimizer();
	inline double get_loss(double* weights = nullptr) const {
		return model->get_loss(weights);
	}
	inline void get_gradient(int idx, double* grad, double* weights = nullptr, int wndim = 1) const {
		return model->get_gradient(idx, grad, weights, wndim);
	}
	inline void get_hess_vt(int idx, double* vt, double* hess_vt, double* weights = nullptr, int vndim = 1) const {
		return model->get_hess_vt(idx, vt, hess_vt, weights, vndim);
	}
	void get_full_gradient(double* grad, double* weights = nullptr, int wndim = 1) {
		return model->get_full_gradient(grad, weights, wndim);
	}  
	
	double log_iter_end(chrono::time_point<chrono::high_resolution_clock> & start, double data_pass, Records& records);
	void log_start() {
		LOG(INFO) << "----------------------" << optimizer_name << "  start------------------";
		params.log_params();
	};
	void log_end(const Records& records) {
		records.log_optimization_end();
		LOG(INFO) << "----------------------" << optimizer_name << "  end------------------";
	};


	void step(double step_size, double* data);
	Model* model;
	double step_size;
	double lambda;
	std::string optimizer_name;
	size_t N;
	Params params;
	bool compare_hess;
};


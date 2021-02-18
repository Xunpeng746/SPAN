#pragma once
#include"config.h"
#include "Model.h"
#include "Lissa.h"
#include "Slbfgs.h"
#include "NewSamp.h"
#include "Svrg.h"
#include "Lant.h"
#include "Optimizer.h"
#include "Logistic.h"
#include "LSvm.h"
#include "Param.hpp"
#include "GD.h"
#include "Sbbfgs.h"
#include <set>
#include<string>
#include <random>
#include <ctime>
#include <cstdlib>
#include <locale> 
#include "json.hpp"
extern json params_json;
class Factory
{
public:
	void get_choice(const vector<double>& vt, int& idx, double& param) {
		param = vt[idx%vt.size()];
		idx /= vt.size();
	}
	void get_choice(const vector<int>& vt, int& idx, int& param) {
		param = vt[idx%vt.size()];
		idx /= vt.size();
	}
	Factory() {
		model = nullptr;
	}
	Factory(std::string model_name, double lambda=1e-4) {
		model = nullptr;
		set_model(model_name, lambda);
		N = model->N;
	};
	Params get_params(std::string optimizer_name, bool rand_flag=false);
	Model* set_model(std::string model_name, double lambda = 1e-4);
	Model* get_model(std::string model_name, double lambda = 1e-4);
	Optimizer* get_solver(std::string optimizer_name, const Params& params);
	~Factory();

	Model* model;
	int N;

};


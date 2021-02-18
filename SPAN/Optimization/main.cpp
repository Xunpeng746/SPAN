
#pragma warning( disable : 4996) 
#include <iostream>
#include <vector>
#include <cstdlib>
#include <time.h>
#include "config.h"
#include "Util.h"
#include "Model.h"
#include "Logistic.h"
#include "TestUtil.h"
#include "Context.hpp"
#include "Factory.h"
#include<mkl.h>

#include "getopt.hpp"
using namespace std;
using json = nlohmann::json;
extern json params_json;
INITIALIZE_EASYLOGGINGPP
extern size_t THREAD_NUM_MKL;
extern size_t THREAD_NUM_OMP;
const int LOOP_COUNT = 1;
string save_path = "./";
const vector<string> ALL_OPT{"lissa","newsamp","slbfgs","sbbfgs","svrg","lant" };
void print(string msg) {
	LOG(INFO) << msg;
}

int run(Context* context ,string solver_name, int max_iter_num = 40, double* init_weight = nullptr) {
	if (context == nullptr) {
		throw "Context is null";
	}
	if (params_json.count(solver_name)==0) {
		printf("there is no %s params",solver_name.c_str());
		throw "there is no " + solver_name + " params";
	}
	Params params(params_json[solver_name]);
	Optimizer* optimizer = context->get_solver(solver_name, params);
	Records records = optimizer->run(max_iter_num, Records(), init_weight);
	records.log_best();
	string filename = save_path + solver_name +"_"+ (optimizer->get_params()).to_string() + ".csv";
	records.save(filename);
	print("save as " + filename);

	delete optimizer;
}


int run_all(string model_name, double lambda = 1e-4, vector<string> optimizers_list = {}, 
	int max_iter_num = 40, int warm_up=1, int init_iter=2) {
	if (optimizers_list.size() == 0) {
		optimizers_list = vector<string>{ "lant","newsamp","slbfgs","sbbfgs","svrg","lissa","lant"};
		// optimizers_list = vector<string>{ "lant","newsamp","lant","lissa","svrg"};
	}
	Context* context = new Context(model_name, lambda);
	context->factory->model->set_use_mm(true);
	int N = context->N;

	//warm start
	double* init_weight = Util::get_randn(MAX_DIM, 0, 1);
	printf("warm_up%d\n",warm_up);
	if(warm_up==1){
		Params params;
		double step_size = params_json["svrg"]["step_size"];
		if(step_size<0.1) step_size=0.1;
		params.set_param("m", N*2).set_param("step_size", step_size);
		Optimizer* svrg = context->get_solver("svrg", params);
		svrg->run(init_iter, Records());
		mem_copy(MAX_DIM, svrg->model->get_model(), 1, init_weight, 1);
		string init_model = save_path+model_name+"_"+to_string(*(context->factory->model->m_lambda))+".model";
		printf("%s\n",init_model.c_str());
		Util::to_csv2(init_model, init_weight, 1, MAX_DIM, MAX_DIM);
		delete svrg;
	}else if(warm_up>=2){
		string init_model = save_path+model_name+"_"+to_string(*(context->factory->model->m_lambda))+".model";
		Util::read_csv(init_model, init_weight, 1, MAX_DIM, MAX_DIM);
	}else{
		mem_zero(MAX_DIM, init_weight);
	}
	//run all optimizer in list
	for (auto optimizer_name : optimizers_list) {
		run(context, optimizer_name, max_iter_num, init_weight);
	}

	mkl_free(init_weight);
	delete context;
	return 0;
}



int main(int argc, char** argv) {
	START_EASYLOGGINGPP(argc, argv);
	std::string params_file = getarg("", "--params-file");
	if(params_file!=""){
		std::ifstream ss(params_file);
		ss >> params_json;
		ss.close();
	}
	
	json default_config = params_json["config"];
	int warm_up = getarg(default_config["warm-up"], "--warm-up");
	int init_iter = getarg(default_config["init-iter"], "--init-iter");
	PRINT_HESS = getarg(default_config["print-hess"], "--print-hess");
	std::string solver_name = getarg("none", "--solver");
	std::string model_name = getarg(default_config["model"], "--model");
	save_path = getarg("./", "--save");
	double lambda = getarg(default_config["l2"], "--l2");
	int search_num = getarg(default_config["search-num"], "--search-num");
	int max_iter_num = getarg(default_config["max-iter"], "--max-iter");
	int _THREAD_NUM = getarg(default_config["t"], "-t");

	
	
	THREAD_NUM_MKL = _THREAD_NUM;
	THREAD_NUM_OMP = 5;
	omp_set_num_threads(THREAD_NUM_OMP);
	mkl_set_num_threads(THREAD_NUM_MKL);

	if (search_num <= 0) {
		if (solver_name == "all") {
			run_all(model_name, lambda,{}, max_iter_num, warm_up,init_iter);
		}else if(solver_name=="none"){
			vector<string> solver_list = default_config["solver"];
			run_all(model_name, lambda, solver_list, max_iter_num, warm_up,init_iter);
		}else{
			run_all(model_name, lambda, {solver_name}, max_iter_num, warm_up,init_iter);
		}
	}else {
		Context* context = new Context(model_name, lambda);
		context->factory->model->set_use_mm(true);
		context->get_best_param(solver_name, search_num, max_iter_num, 1);
		delete context;
	}
	return 0;
}
#include "config.h"
#include "Factory.h"
#include "Util.h"
#include "easylogging++.h"
extern size_t MAX_DIM;

class Context {
public:
#if defined(__linux__)
	// Linux系统
	string WORKSTATION_PATH = "./";
#elif defined(_WIN32)
	// Windows系统
	string WORKSTATION_PATH = "./";
#endif	
	Context(std::string model_name, double lambda = 1e-4) {
		factory = new Factory(model_name, lambda);
		N = factory->model->N;
	}
	void log_msg(std::string msg) {
		LOG(INFO) << msg;
	}
	Optimizer* get_solver(std::string optimizer_name, const Params& params) {
		return factory->get_solver(optimizer_name, params);
	}
	void log_test_start(std::string optimizer_name, int test_count) {
		LOG(INFO) << "--------------"<<optimizer_name<<"  "<<to_string(test_count)<<" test start-------------";
	}
	void log_test_end(std::string optimizer_name, int test_count) {
		LOG(INFO) << "--------------" << optimizer_name << "  " << to_string(test_count) << " test end-------------";
	}
	void get_best_param(std::string optimizer_name, int test_num, int max_epoch_num, int warm_start=1) {
		Records best_records;

		std::string save_file_name = WORKSTATION_PATH + optimizer_name + "_" + to_string(test_num) +
			"_" + to_string(factory->model->m_lambda[0]) + ".csv";
		printf("filename:%s\n", save_file_name.c_str());
		double* init_weights = Util::get_randn(MAX_DIM, 0, 1);
		Records init_records;
		log_msg("warm start"+to_string(warm_start));
		if (warm_start>0) {
			Params params;
			params.set_param("step_size", 0.1).set_param("m", 2 * N);
			params.log_params();
			Optimizer* svrg = (Optimizer*)get_solver("svrg", params);
			init_records = svrg->run(3, init_records, init_weights);
			mem_copy(MAX_DIM, factory->model->get_model(), 1, init_weights, 1);
			delete svrg;
		}
		//需要将预热init_record中记录加入到实际运行的记录中
		for (int i = 0; i < test_num; ++i) {
			log_test_start(optimizer_name, i);
			Params params = factory->get_params(optimizer_name, false);
			Optimizer* solver = (Optimizer*)get_solver(optimizer_name, params);\
			solver->params.log_params();
			Records now_records;
			now_records = solver->run(max_epoch_num, now_records, init_weights);
			if (now_records.best_record < best_records.best_record) {
				best_records = now_records;
				best_records.log_best();
			}
			log_test_end(optimizer_name, i);
			delete solver;
		}
		best_records.log_best();
		best_records.save(save_file_name);
		mkl_free(init_weights);
	}

	~Context() {
		delete factory;
	}

	Factory* factory;
	int N;
};
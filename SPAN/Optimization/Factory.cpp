#include "Factory.h"

Params Factory::get_params(std::string optimizer_name, bool rand_flag)
{
	static unsigned int count = -1;
	count += 1;
	if (rand_flag) {
		srand((unsigned)time(0));
		count = rand();
	}
	int idx = count;
	if (optimizer_name == "lissa") {
		Params params;
		json params_list = params_json["lissa"];
		params.set_choice_param("s2", params_list["s2"], idx);
		params.set_choice_param("s1", params_list["s1"], idx);
		params.set_choice_param("step_size", params_list["step_size"], idx);
		params.set_param("t1", 1);
		// params.set_choice_param("t1", params_list["t1"], idx);
		return params;
	}
	else if (optimizer_name == "newsamp") {
		Params params;
		json params_list = params_json["newsamp"];
		params.set_choice_param("step_size", params_list["step_size"], idx);
		int rank = params.set_choice_param("rank",  params_list["rank"], idx);
		params.set_choice_param("sample_size", params_list["sample_size"], idx);
		return params;
	}
	else if (optimizer_name == "slbfgs") {
		Params params;
		json params_list = params_json["slbfgs"];
		double b = params.set_choice_param("b",params_list["b"], idx);
		params.set_choice_param("bh", params_list["bh"], idx, b);
		params.set_choice_param("step_size", params_list["step_size"], idx);
		double update_period = 10, store_num = 10, m = N / b;
		params.set_param("update_period", update_period);
		params.set_param("store_num", store_num);
		params.set_param("m", m);
		return params;
	}
	else if (optimizer_name == "svrg") {
		Params params;
		json params_list = params_json["svrg"];
		params.set_choice_param("step_size", params_list["step_size"], idx);
		params.set_choice_param("m", {2}, idx, N);
		return params;
	}
	else if (optimizer_name == "gd") {

	}
	else if (optimizer_name == "sbbfgs") {
		Params params;
		json params_list = params_json["sbbfgs"];
		double b = params.set_choice_param("b", { 1}, idx, (int)sqrt(N));
		params.set_choice_param("step_size", params_list["step_size"], idx);
		params.set_choice_param("rank", params_list["rank"], idx);
		params.set_choice_param("store_num", params_list["store_num"], idx);
		params.set_choice_param("bh", params_list["bh"], idx, b);
		params.set_param("m", int(N / b));
		return params;
	}
	else if (optimizer_name == "lant") {
		Params params;
		json params_list = params_json["lant"];
		
		int keep_rank = params.set_choice_param("keep_rank", params_list["keep_rank"], idx);
		params.set_choice_param("step_size", params_list["step_size"],  idx);
		params.set_choice_param("scale_w", params_list["scale_w"], idx);
		params.set_choice_param("more_rank", params_list["more_rank"], idx);
		params.set_choice_param("power_iter", params_list["power_iter"], idx);
		params.set_choice_param("sample_size", params_list["sample_size"], idx);
		params.set_param("stage_a", 3);
		params.set_param("stage_b", 3);
		params.set_param("use_hess_vt", 1);
		return params;
	}

}

Model * Factory::set_model(std::string model_name, double lambda)
{
	if (model != nullptr) {
		delete model;
		model = nullptr;
	}
	if(model_name == "LR49"){
		if(lambda >= 1e-3){
			min_loss = LR49_3_loss;
		}else{
			min_loss = LR49_4_loss;
		}
		eps = 1e-14;
	}else if(model_name=="LRCT"){
		min_loss = LRCT_loss;
	}else if(model_name=="LRSynthetic"){
		min_loss = LRSynthetic_loss;
	}else if(model_name == "LRCovtype"){
		if (abs(lambda - 1e-6) <= 1e-10) {
			min_loss = LRCOVTYPE_6_LOSS;
		}
		else if(abs(lambda - 1e-4) <= 1e-10){
			min_loss = LRCOVTYPE_7_LOSS;
			eps = 1e-7;
		}
		else if (abs(lambda - 1e-5) <= 1e-10) {
			min_loss = LRCOVTYPE_5_LOSS;
			//eps = 1e-9;
		}
	}
	else if (model_name == "MSD") {

	}
	else if (model_name == "LSvmWebSpam") {
		if (abs(lambda - 1e-6) <= 1e-10) {
			min_loss = WEBSPAM_6_LOSS;
			eps = 1e-12;
		}
		else if (abs(lambda - 1e-5) <= 1e-10) {
			min_loss = WEBSPAM_5_LOSS;
			eps = 1e-11;
		}
		else if (abs(lambda - 1e-4) <= 1e-10) {
			min_loss = WEBSPAM_4_LOSS;
			eps = 1e-10;
		}
	}
	else if (model_name == "LSvmYELP") {
		if (abs(lambda - 1e-6) <= 1e-10) {
			min_loss = 0.9400316848028231;
			eps = 1e-10;
		}
		else if (abs(lambda - 1e-5) <= 1e-10) {
			min_loss = 0.9314994474595044;
			min_loss = 0.9456952100180807;
			eps = 1e-10;
		}
		else if (abs(lambda - 1e-4) <= 1e-10) {
			min_loss = 0.9508199885459072;
			eps = 1e-10;
		}
	}
	else {
		printf("does not exist this model %s\n" , model_name.c_str());
	}
	model = get_model(model_name, lambda);
	N = model->N;
	return model;
}

Model * Factory::get_model(std::string model_name, double lambda)
{
	double *X = nullptr, *Y = nullptr;
	int N = 0;
	if (model_name.find("49") != string::npos) {
		N = Util::loadMnist49(&X, &Y);
	}else if(model_name.find("Synthetic") != string::npos) {
		N = Util::loadLRSynthetic(&X, &Y);
	}else if (model_name.find("Covtype") != string::npos) {
		N = Util::loadLRCovtype(&X, &Y);
	}else if (model_name.find("WebSpam") != string::npos) {
		N = Util::loadWebSpam(&X, &Y);
	}else if (model_name.find("YELP") != string::npos) {
		N = Util::loadYELP(&X, &Y);
	}else{
		printf("does not exist this dataset %s\n" , model_name.c_str());
	}
	Model* _model = nullptr;
	if(model_name.find("Svm") != string::npos){
		_model = new LSvm(X, Y, N, lambda);
	}else{
		_model = new Logistic(X, Y, N, lambda);
	}
	return _model;
}

Optimizer * Factory::get_solver(std::string optimizer_name, const Params & params)
{
	if (model == nullptr) {
		throw "model null";
	}
	Optimizer* optimizer = nullptr;
	if (optimizer_name == "lissa") {
		optimizer = new Lissa(model, params);
	}else if (optimizer_name == "newsamp") {
		optimizer = new NewSamp(model, params);
	}else if (optimizer_name == "slbfgs") {
		optimizer = new Slbfgs(model, params);
	}else if (optimizer_name == "svrg") {
		optimizer = new Svrg(model, params);
	}else if (optimizer_name == "gd") {
		optimizer = new GD(model, params);
	}else if (optimizer_name == "sbbfgs") {
		optimizer = new Sbbfgs(model, params);
	}else if (optimizer_name == "lant") {
		optimizer = new Lant(model, params);
	}else {
		throw "does exist such optimizer " + optimizer_name;
	}
	return optimizer;
}


Factory::~Factory()
{
	delete model;
}

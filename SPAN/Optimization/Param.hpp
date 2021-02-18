#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <set>
#include<string>
#include<map>
#include<vector>
#include "json.hpp"
#include "easylogging++.h"
using json = nlohmann::json;
class Params {


public:
	Params() {}
	Params(const json& _params ) {
		for (auto iter : _params.items()) {
			string param_name = iter.key();
			double param_value = iter.value();
			set_param(param_name, param_value);
		}
	}
	Params(const Params& _params) {
		params_dict_double = _params.params_dict_double;
		params_dict_int = _params.params_dict_int;
	}
	std::string to_string(string sep1="_", string sep2 = "_") {
		std::string msg;
		for (auto v : params_dict_double) {
			msg += v.first + sep1 + std::to_string(v.second) + sep2;
		}
		for (auto v : params_dict_int) {
			msg += v.first + sep1 + std::to_string(v.second) + sep2;
		}
		return msg;
	}
	void log_params() {
		std::string msg = to_string(":", " ");
		LOG(INFO) << msg;
	};

	double get_param(std::string param_name) {
		if (params_dict_double.count(param_name)) {
			return params_dict_double[param_name];
		}else if (params_dict_int.count(param_name)) {
			return params_dict_int[param_name];
		}
		return 0;
	}

	Params& set_param(std::string param_name, double param_value) {
		if (ceil(param_value) == floor(param_value)){
			params_dict_int[param_name] = param_value;
		}else {
			params_dict_double[param_name] = param_value;
		}
		return *this;
	}
	Params& clear() {
		params_dict_double.clear();
		params_dict_int.clear();
		return *this;
	}

	double set_choice_param(std::string param_name, std::vector<double> param_list, int& param_idx, double base=1) {
		if (param_list.size() == 0) return param_idx;
		double param_value = param_list[param_idx%param_list.size()];
		set_param(param_name, param_value*base);
		param_idx /= param_list.size();
		return param_value * base;
	}
private:
	std::map<std::string, double> params_dict_double;
	std::map<std::string, int> params_dict_int;
};

#endif PARAMS_HPP
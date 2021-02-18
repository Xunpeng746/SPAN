#pragma once
#ifndef RECORDS_HPP
#define RECORDS_HPP
#include<vector>
#include<iostream>
#include<fstream>
#include <iomanip>
#include "config.h"
//#include <algorithm>
const double loss_eps = 2e-15;
const double time_eps = 1e-2;
extern double min_loss;
extern double eps;
class Record {
public:
	double time;
	double data_pass;
	double loss;
	double hess_error;
	friend class Records;
	Record() {
		loss = 1e9;
		data_pass = 1e9;
		time = 1e9;
	};
	Record(double _time, double _iter, double _loss, double _hess_error=1) {
		time = _time;
		data_pass = _iter;
		loss = _loss;
		hess_error = _hess_error;
	}
	bool operator <(const Record& rhs) {
		if((loss < min_loss+eps)&&(rhs.loss < min_loss+eps)){
			//add code to decide which record is better
			return data_pass < rhs.data_pass;
			// return (time*5+data_pass<rhs.time*5+rhs.data_pass);
		}
		return (loss < (rhs.loss - loss_eps));
	}
	void print() {
		char msg[100];
		loss = min(loss, 1e10);
		sprintf(msg, "time:%.2f  iter_num:%.2f  loss:%.16f hess_error%.10f", time, data_pass, loss, hess_error);
		LOG(INFO) << msg;
	}
	Record operator+(const Record& rhs){
		Record ans;
		ans.time = time + rhs.time;
		ans.data_pass = data_pass + rhs.data_pass;
		ans.loss = loss + rhs.loss;
		ans.hess_error = hess_error + rhs.hess_error;
		return ans;
	} 
	Record operator/(const int& div_num){
		Record ans;
		ans.time = time/div_num;
		ans.data_pass = data_pass/div_num;
		ans.loss = loss/div_num;
		ans.hess_error = hess_error / div_num;
		return ans;
	} 
};
class Records {
public:
	std::vector<Record> record_list;
	Record best_record;
	void save(std::string filename) {
		std::ofstream out(filename);
		out << std::setprecision(8);
		char loss[50];
		int i = 0;
		for (auto r : record_list) {
			++i;
			double rloss = r.loss;
			rloss = mi(rloss, 100);
			if (rloss < -100) rloss = -100;
			sprintf(loss, "%.16f", r.loss);
			out << r.data_pass << "," << r.time << "," << loss << ","<<r.hess_error<< std::endl;
		}
	}
	Records operator+(const Records& rhs){
		if(record_list.size() != rhs.record_list.size()){
			throw "records size not equal";
		}
		Records ans(*this);
		for(int i = 0; i < ans.record_list.size(); ++i){
			ans.record_list[i] = ans.record_list[i] + rhs.record_list[i];
		}
		return ans;
	} 
	Records operator/(const int& div_num){
		Records ans(*this);
		for(int i = 0; i < ans.record_list.size(); ++i){
			ans.record_list[i] = ans.record_list[i]/div_num;
		}
		return ans;
	} 
	void reset_best(){
		best_record = Record();
		for(int i = 0; i < record_list.size(); ++i){
			best_record = mi(best_record, record_list[i]);	
		}
	}

	void set_hess_error(double hess_error) {
		int len = record_list.size();
		if (len < 1) return;
		record_list[len - 1].hess_error = hess_error;
	}

	void push_back(const Record& record) {
		record_list.push_back(record);
		if(record.loss>0)
		best_record = mi(best_record, record);
	}
	void log_optimization_end()const {
		char loss[50];
		sprintf(loss, "%.16f", best_record.loss);
		LOG(INFO) << "this optimization iter_num is:" << best_record.data_pass;
		LOG(INFO) << "this optimization costTime is:" << best_record.time;
		LOG(INFO) << "this optimization loss is:" << loss;
	};
	void log_best() const{
		char loss[50];
		sprintf(loss, "%.16f", best_record.loss);
		LOG(INFO) << "now best iter_num is:" << best_record.data_pass;
		LOG(INFO) << "now best costTime is:" << best_record.time;
		LOG(INFO) << "now best loss is:" << loss;
	};

};
#endif RECORDS_HPP

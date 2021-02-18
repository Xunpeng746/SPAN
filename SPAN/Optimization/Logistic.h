
#pragma once


#ifndef LOGISTIC_H
#define LOGISTIC_H
#include "Model.h"
#include <cmath>
#include <algorithm>


class Logistic :
	public Model
{
public:
	~Logistic(){
		mkl_free(ones);
	}
	Logistic(double* _X, double* _Y, size_t N, double param) :Model(_X, _Y, N, param) {
		ones = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
		for(int i = 0; i < MAX_DIM; ++i){
			ones[i] = 1;
		}
	};
	double* ones;
	// int classfify(const double* sample)const;
	double get_loss(const double* weights = nullptr) const;
	void get_gradient(int idx, double* grad, const double* weights = nullptr, int wndim = 1, bool add=0) ;
	void get_gradient_mm(double* data_x, double* data_y, int n_samples, double* grad, const double* weights = nullptr, int wndim = 1);
	void get_hess_vt(int idx, const double* vt, double* hess_vt, const double* weights = nullptr, int vndim = 1, bool add = 0) ;
	void get_hessian(const int idx, double* hess, const double* weights = nullptr, bool add = false);

	void get_hess_vt_mm(double* data_x, double* data_y, int n_samples, const double* vt, double* hess_vt, 
		const double* weights = nullptr, int vndim = 1, int _power_iter=-1, double* B=nullptr) ;
	void get_hessian_mm(double* data_x, double* data_y, int n_samples, double* hess, const double* weights = nullptr) ;
};

 

#endif // !LOGISTIC_H


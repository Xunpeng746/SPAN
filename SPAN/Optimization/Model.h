#pragma once
#ifndef MODEL_H
#define MODEL_H
#include <string>
#include <set>
#include "config.h"
#include "Util.h"
extern size_t MAX_DIM;

class Model
{
public:
	Model() {}
	Model(double* _X, double* _Y, size_t _N, double _param) {
		X = _X;
		Y = _Y;
		N = _N;
		m_lambda = new double;
		*m_lambda = _param;
		m_weights = (double*)mkl_malloc(sizeof(double)*MAX_DIM, 64);
		use_mm = true;
	}
	virtual ~Model() {
		delete m_lambda;
		mkl_free(m_weights);
	}
	void set_use_mm(bool _use_mm) {
		use_mm = _use_mm;
	}

	virtual double get_loss(const double* weights = nullptr) const=0;
	virtual void get_gradient(int idx, double* grad, const double* weights = nullptr, int wndim = 1, bool add = 0)=0;
	virtual void get_hess_vt(int idx,const double* vt, double* hess_vt, const double* weights = nullptr, int vndim = 1, bool add = 0)=0;
	virtual void get_hessian(const int idx,double* hess, const double* weights = nullptr, bool add = false) =0;
	virtual void get_gradient_mm(double* data_x, double* data_y, int n_samples, double* grad, const double* weights = nullptr, int wndim = 1) = 0;
	virtual void get_hess_vt_mm(double* data_x, double* data_y, int n_samples, const double* vt, double* hess_vt, 
		const double* weights = nullptr, int vndim = 1, int _power_iter=-1, double* B=nullptr) = 0;
	// virtual void get_hess_vt_mm_k(double* data_x, double* data_y, int n_samples, const double* vt, double* hess_vt, const double* weights = nullptr, int vndim = 1, int k=1) = 0;
	virtual void get_hessian_mm(double* data_x, double* data_y, int n_samples, double* hess, const double* weights = nullptr) = 0;

	//以下无需继承, 上述基础操作实现完之后, Model下列操作只需要调用上述
	void get_gradient(const int idx[], int n_samples,double* grad,const double* weights = nullptr, int wndim = 1) ;
	void get_hess_vt(const int idx[], int n_samples,const double* vt, double* hess_vt,
		const double* weights = nullptr, int vndim = 1, int _power_iter=-1, double* B=nullptr) ;
	void get_hessian(const int idx[], int n_samples, double* hess, const double* weights = nullptr);
	void get_full_gradient(double* grad, const double* weights = nullptr, int wndim = 1);
	 void get_full_hessian(double* hess, const double* weights = nullptr);
	void set_init_weight(const double* weights);
	double* get_model() const;
	void init_model(double* weights=nullptr);
	double* get_params() const;
	void step(double step_size, double* data);
	void update_model(const double* new_weights);
	void update_model_by_add(const double* grad);

//protected:
	double* m_lambda;
	double* m_weights;
	double *X;
	double *Y;
	size_t N;
	bool use_mm;

};


#endif


#pragma warning( disable : 4996) 
#include<cstdio>
#include<cstdlib>
#include<string>
#include<utility>
#include <iostream>
#include"config.h"
#include <ctime>
#include<vector>
#include<chrono>
#include<fstream>
#include <iomanip>
 extern size_t MAX_DIM;
// extern size_t DATA_DIM;
typedef std::pair<int, int> Pair;
#include "MemFactory.hpp"




namespace Util {

	double spectral_norm(double* data, int n);
	double norm(double* data, int n);
	void to_csv(string filename, double * data, int m, int n, int lda);
	void to_csv2(string filename, double * data, int m, int n, int lda);
	void read_csv(string filename, double* data, int m, int n, int lda);
	Pair loadData(double** data, std::string path);
	Pair fast_loadData(double** data, std::string path);
	int loadStdData(double** X, double** Y, string path_x, string path_y);
	int fast_loadStdData(double** X, double** Y, string path_x, string path_y);
	int loadMnist49(double** X, double** Y);
	int loadLRCT(double** X, double** Y);
	int loadLRCovtype(double** X, double** Y);
	int loadWebSpam(double** X, double** Y);
	int loadMSD(double** X, double** Y);
	int loadYELP(double** X, double** Y);
	int loadLRSynthetic(double** X, double** Y);
	void get_randn(double* data, int num, double mu, double sigma);
	void get_rand_choice(int* data, int num, int l, int r);
	int ksvd(double* data, double* u, double* s, int n, int nselect);
	void inv(double* data, int n);
	double* get_randn(int num, double mu, double sigma);
	int* get_rand_choice(int num, int l, int r);
	void print_matrix(std::string name, const double* a, int M, int N, int lda);
}

#include "config.h"


double eps = 1e-12;
size_t MAX_DIM = 100;
size_t DATA_DIM = 100;
size_t THREAD_NUM_MKL = 5;
size_t THREAD_NUM_OMP = 5;
double min_loss;
bool PRINT_HESS = true;
double delta_lambda=1;
json params_json;


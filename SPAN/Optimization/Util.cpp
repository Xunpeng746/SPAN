#include "Util.h"
extern size_t MAX_DIM;

#if defined(__linux__)
// Linux
const string DATA_PATH = "..//..//LR//resource//";
#elif defined(_WIN32)
// Windows
const string DATA_PATH = "./";
#endif


namespace fastIO{
    #define BUF_SIZE 100000
    #define OUT_SIZE 100000
    #define ll long long
    //fread->read
    bool IOerror=0;
    inline char nc(){
        static char buf[BUF_SIZE],*p1=buf+BUF_SIZE,*pend=buf+BUF_SIZE;
        if (p1==pend){
            p1=buf; pend=buf+fread(buf,1,BUF_SIZE,stdin);
            if (pend==p1){IOerror=1;return -1;}
            //{printf("IO error!\n");system("pause");for (;;);exit(0);}
        }
        return *p1++;
    }
    inline bool blank(char ch){return ch==' '||ch=='\n'||ch=='\r'||ch=='\t';}
    inline void read(int &x){
        bool sign=0; char ch=nc(); x=0;
        for (;blank(ch);ch=nc());
        if (IOerror)return;
        if (ch=='-')sign=1,ch=nc();
        for (;ch>='0'&&ch<='9';ch=nc())x=x*10+ch-'0';
        if (sign)x=-x;
    }
    inline void read(ll &x){
        bool sign=0; char ch=nc(); x=0;
        for (;blank(ch);ch=nc());
        if (IOerror)return;
        if (ch=='-')sign=1,ch=nc();
        for (;ch>='0'&&ch<='9';ch=nc())x=x*10+ch-'0';
        if (sign)x=-x;
    }
    inline void read(double &x){
        bool sign=0; char ch=nc(); x=0;
        for (;blank(ch);ch=nc());
        if (IOerror)return;
        if (ch=='-')sign=1,ch=nc();
        for (;ch>='0'&&ch<='9';ch=nc())x=x*10+ch-'0';
        if (ch=='.'){
            double tmp=1; ch=nc();
            for (;ch>='0'&&ch<='9';ch=nc())tmp/=10.0,x+=tmp*(ch-'0');
        }
        if (sign)x=-x;
    }
    inline void read(char *s){
        char ch=nc();
        for (;blank(ch);ch=nc());
        if (IOerror)return;
        for (;!blank(ch)&&!IOerror;ch=nc())*s++=ch;
        *s=0;
    }
    inline void read(char &c){
        for (c=nc();blank(c);c=nc());
        if (IOerror){c=-1;return;}
    }
    //fwrite->write
    struct Ostream_fwrite{
        char *buf,*p1,*pend;
        Ostream_fwrite(){buf=new char[BUF_SIZE];p1=buf;pend=buf+BUF_SIZE;}
        void out(char ch){
            if (p1==pend){
                fwrite(buf,1,BUF_SIZE,stdout);p1=buf;
            }
            *p1++=ch;
        }
        void print(int x){
            static char s[15],*s1;s1=s;
            if (!x)*s1++='0';if (x<0)out('-'),x=-x;
            while(x)*s1++=x%10+'0',x/=10;
            while(s1--!=s)out(*s1);
        }
        void println(int x){
            static char s[15],*s1;s1=s;
            if (!x)*s1++='0';if (x<0)out('-'),x=-x;
            while(x)*s1++=x%10+'0',x/=10;
            while(s1--!=s)out(*s1); out('\n');
        }
        void print(ll x){
            static char s[25],*s1;s1=s;
            if (!x)*s1++='0';if (x<0)out('-'),x=-x;
            while(x)*s1++=x%10+'0',x/=10;
            while(s1--!=s)out(*s1);
        }
        void println(ll x){
            static char s[25],*s1;s1=s;
            if (!x)*s1++='0';if (x<0)out('-'),x=-x;
            while(x)*s1++=x%10+'0',x/=10;
            while(s1--!=s)out(*s1); out('\n');
        }
        void print(double x,int y){
            static ll mul[]={1,10,100,1000,10000,100000,1000000,10000000,100000000,
                1000000000,10000000000LL,100000000000LL,1000000000000LL,10000000000000LL,
                100000000000000LL,1000000000000000LL,10000000000000000LL,100000000000000000LL};
            if (x<-1e-12)out('-'),x=-x;x*=mul[y];
            ll x1=(ll)floor(x); if (x-floor(x)>=0.5)++x1;
            ll x2=x1/mul[y],x3=x1-x2*mul[y]; print(x2);
            if (y>0){out('.'); for (size_t i=1;i<y&&x3*mul[i]<mul[y];out('0'),++i); print(x3);}
        }
        void println(double x,int y){print(x,y);out('\n');}
        void print(char *s){while (*s)out(*s++);}
        void println(char *s){while (*s)out(*s++);out('\n');}
        void flush(){if (p1!=buf){fwrite(buf,1,p1-buf,stdout);p1=buf;}}
        ~Ostream_fwrite(){flush();}
    }Ostream;
    inline void print(int x){Ostream.print(x);}
    inline void println(int x){Ostream.println(x);}
    inline void print(char x){Ostream.out(x);}
    inline void println(char x){Ostream.out(x);Ostream.out('\n');}
    inline void print(ll x){Ostream.print(x);}
    inline void println(ll x){Ostream.println(x);}
    inline void print(double x,int y){Ostream.print(x,y);}	//y为小数点后几位
    inline void println(double x,int y){Ostream.println(x,y);}
    inline void print(char *s){Ostream.print(s);}
    inline void println(char *s){Ostream.println(s);}
    inline void println(){Ostream.out('\n');}
    inline void flush(){Ostream.flush();}			//清空
    #undef ll
    #undef OUT_SIZE
    #undef BUF_SIZE
};

using namespace fastIO;
double Util::spectral_norm(double* data, int n){

	mkl_set_num_threads_local(1);
	double* data_tmp = (double*)mkl_malloc(sizeof(double)*n*n, 64);
	gemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, n, 1.0, data, n, data, n, 0, data_tmp, n);

	int* isppz = (int*)mkl_malloc(sizeof(int)*n, 64);
	int il = 0, iu = 0, m = 0;
	double abstol = -1, vl = 0, vu = 0;
	double u = 0, s = 0;
	il = n - 1 + 1, iu = n;
	int nselect = 1;
	int ret = LAPACKE_dsyevr(LAPACK_ROW_MAJOR, 'N', 'I', 'U', n, data_tmp, n,
		vl, vu, il, iu, abstol, &m, &s, &u, nselect, isppz);
	mkl_set_num_threads_local(0);
	if (ret != 0) {
		cout << "norm error ret:" << ret << endl;
		// throw "norm error";
	}
	mkl_free(isppz);
	mkl_free(data_tmp);
	double ans = sqrt(s);
	return ans;
}

double Util::norm(double * data, int n)
{

	// int more_rank = 20;
	// int rank = min(more_rank,n);
	// double* q_matrix = (double*)mkl_malloc(sizeof(double)*n*rank, 64);
	// double* B = (double*)mkl_malloc(sizeof(double)*rank*rank, 64);
	// double* tau = (double*)mkl_malloc(sizeof(double)*n, 64);
	// double* s = (double*)mkl_malloc(sizeof(double)*n, 64);
	// double* hess_dot_matrix = (double*)mkl_malloc(sizeof(double)*n*rank*2, 64);
	// double* mat[2];
	// mat[0] = hess_dot_matrix, mat[1] = hess_dot_matrix + n * rank;
	// Util::get_randn(mat[0], n*rank, 0,1);

	// for (int i = 0; i < 7; ++i) {
	// 	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rank, n, n, 1.0, mat[0], n, data, n, 0, mat[1], n);
	// 	 LAPACKE_dgelqf(CblasRowMajor, rank, MAX_DIM, mat[1], n, tau);
	// 	 LAPACKE_dorglq(CblasRowMajor, rank, MAX_DIM, rank, mat[1], n, tau);
	// 	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rank, n, n, 1.0, mat[1], n, data, n, 0, mat[0], n);
	// 	LAPACKE_dgelqf(CblasRowMajor, rank, MAX_DIM, mat[0], n, tau);
	// 	LAPACKE_dorglq(CblasRowMajor, rank, MAX_DIM, rank, mat[0], n, tau);
	// }
	// cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rank, n, n, 1.0, mat[0], n, data, n, 0, mat[1], n);
	// LAPACKE_dgelqf(CblasRowMajor, rank, n, mat[1], n, tau);
	// LAPACKE_dorglq(CblasRowMajor, rank, n, rank, mat[1], n, tau);

	// cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rank, n, n, 1.0, mat[1], n, data, n, 0, mat[0], n);

	// cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rank, rank, n, 1.0, mat[1], n, mat[0], n, 0, B, rank);

	// LAPACKE_dsyevd(CblasRowMajor, 'V', 'U', rank, B, rank, s);

	// //gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rank, MAX_DIM, rank, 1.0, B, rank, q_matrix, MAX_DIM, 0, u, MAX_DIM);

	// double ans = s[rank - 1];
	// mkl_free(q_matrix);
	// mkl_free(B);
	// mkl_free(tau);
	// mkl_free(s);
	// mkl_free(hess_dot_matrix);
	// return ans;
	throw "not implement";
}

void Util::to_csv(string filename, double * data, int m, int n, int lda)
{
	std::ofstream out(filename);
	out << std::setprecision(8);
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			out << data[i*lda + j] << (j != n - 1 ? ',' : '\n');
		}
	}
	out.close();
}
void Util::to_csv2(string filename, double * data, int m, int n, int lda)
{


	std::ofstream out(filename);
	out << std::setprecision(5);	
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			out << data[i*lda + j] << (j != n - 1 ? ' ' : '\n');
		}
	}
	out.close();
}
void Util::read_csv(string filename, double * data, int m, int n, int lda)
{
	std::ifstream in(filename);
	in.setf(in.scientific);
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			in >> data[i*lda + j] ;
		}
	}
	in.close();
}

Pair Util::fast_loadData(double** data, std::string path) {
	freopen((path).c_str(), "r", stdin);
	int k = 0, n = 0;
	read(n);read(k);
	// scanf("%d%d",&n,&k);
	printf("file:%s\n", (path).c_str());
	*data = (double*)mkl_malloc(sizeof(double)*n*k, 64);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < k; ++j) {
			read( *((*data) + i * k + j) );
			// scanf("%lf", (*data) + i * k + j);
			//printf("%.10f", *((*data) + i * k + j));
		}
	}
	std::cout << "path:" << path << "n:" << n << "k:" << k << std::endl;
	fclose(stdin);
	return Pair(n, k);
}

Pair Util::loadData(double** data, std::string path) {
	freopen((path).c_str(), "r", stdin);
	int k = 0, n = 0;
	// read(n);read(k);
	scanf("%d%d",&n,&k);
	printf("file:%s\n", (path).c_str());
	*data = (double*)mkl_malloc(sizeof(double)*n*k, 64);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < k; ++j) {
			// read( *((*data) + i * k + j) );
			scanf("%lf", (*data) + i * k + j);
			//printf("%.10f", *((*data) + i * k + j));
		}
	}
	std::cout << "path:" << path << "n:" << n << "k:" << k << std::endl;
	fclose(stdin);
	return Pair(n, k);
}

int Util::loadMnist49(double** X, double** Y)
{
	int N = loadStdData(X, Y, DATA_PATH + "MNIST49_DATA.in", DATA_PATH + "MNIST49_LABEL.in");
	return N;
}

int Util::loadWebSpam(double** X, double** Y){
	int N = loadStdData(X, Y, DATA_PATH + "WEB_SPAM_DATA.in", DATA_PATH + "WEB_SPAM_LABEL.in");
	return N;
}

int Util::loadLRCovtype(double ** X, double ** Y)
{
	int N = loadStdData(X, Y, DATA_PATH + "Covtype_DATA.in", DATA_PATH + "Covtype_LABEL.in");
	return N;
}
int Util::loadLRSynthetic(double ** X, double ** Y)
{
	int N = loadStdData(X, Y, DATA_PATH + "Synthetic_DATA.in", DATA_PATH + "Synthetic_LABEL.in");
	return N;
}
int Util::loadLRCT(double ** X, double ** Y)
{
	int N = loadStdData(X, Y, DATA_PATH + "LRCT_DATA.in", DATA_PATH + "LRCT_LABEL.in");
	return N;

}


int Util::loadYELP(double** X, double** Y){
	int N = fast_loadStdData(X, Y, DATA_PATH + "yelp_data_r1000.in", DATA_PATH + "yelp_label_r1000.in");
	return N;
}

int Util::loadMSD(double ** X, double ** Y)
{
	int N = loadStdData(X, Y, DATA_PATH + "MSD_DATA.in", DATA_PATH + "MSD_LABEL.in");
	return N;
}

int Util::loadStdData(double ** X, double ** Y, string path_x, string path_y)
{
	Pair res1, res2;
	int N = 0;
	if (path_x == "" || path_y == "") {
		throw "file not exist";
	}
	res1 = Util::loadData(X, path_x);
	res2 = Util::loadData(Y, path_y);
	N = res1.first;
	MAX_DIM = res1.second*res2.second;
	return N;
}

int Util::fast_loadStdData(double ** X, double ** Y, string path_x, string path_y)
{
	Pair res1, res2;
	int N = 0;
	if (path_x == "" || path_y == "") {
		throw "file not exist";
	}
	
	res1 = Util::fast_loadData(X, path_x);
	res2 = Util::fast_loadData(Y, path_y);

	N = res1.first;
	MAX_DIM = res1.second*res2.second;
	return N;
}


void Util::get_randn(double* data, int num, double mu, double sigma) {
	VSLStreamStatePtr stream_MCG31;
	vslNewStream(&stream_MCG31, VSL_BRNG_MCG31, __rdtsc());
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream_MCG31, num, data, mu, sigma);
	vslDeleteStream(&stream_MCG31);
}

void Util::get_rand_choice(int * data, int num, int l, int r)
{
	VSLStreamStatePtr stream_uniform;
	vslNewStream(&stream_uniform, VSL_BRNG_MT19937, __rdtsc());
	randint(num, data, l, r, stream_uniform);
	vslDeleteStream(&stream_uniform);
}

int Util::ksvd(double * data, double * u, double * s, int n, int nselect)
{
	int* isppz = (int*)mkl_malloc(sizeof(int)*n, 64);
	int il = 0, iu = 0, m = 0;
	double abstol = -1, vl = 0, vu = 0;
	il = n-nselect+1, iu = n;
	LAPACKE_dsyevr(LAPACK_ROW_MAJOR, 'V', 'I', 'U', n, data, n,
		vl, vu, il, iu, abstol, &m, s, u, nselect, isppz);
	mkl_free(isppz);
	return m;
}

void Util::inv(double * data, int n)
{
	int* tmp = (int*)mkl_malloc(sizeof(int)*n, 64);
	LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, data, n, tmp);
	LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, data, n, tmp);
	mkl_free(tmp);
}

double* Util::get_randn(int num, double mu, double sigma) {
	double* data = (double*)mkl_malloc(sizeof(double)*num, 64);
	VSLStreamStatePtr stream_MCG31;
	vslNewStream(&stream_MCG31, VSL_BRNG_MCG31, __rdtsc());
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream_MCG31, num, data, mu, sigma);
	vslDeleteStream(&stream_MCG31);
	return data;
}

int * Util::get_rand_choice(int num, int l, int r)
{
	int* idx = (int*)mkl_malloc(sizeof(int)*(num), 64);
	VSLStreamStatePtr stream_uniform;
	vslNewStream(&stream_uniform, VSL_BRNG_MT19937, __rdtsc());
	randint(num, idx, l, r, stream_uniform);
	vslDeleteStream(&stream_uniform);
	return idx;
}

void Util::print_matrix(std::string name, const double * a, int M, int N, int lda)
{
	if (lda <= 0) lda = N;
	std::cout << "----------------------" << std::endl;
	std:: cout << name << std::endl;
	for (int i = 0; i < M; ++i) {
		for(int j = 0; j < N; ++j){
			printf("%.10f    ", a[i*lda + j]);
		}
		printf("\n");
	}
}

//void Util::print_matrix(std::string name,double * a, int M, int N, int lda)
//{
//	if (lda < 0) lda = N;
//	std::cout << "----------------------" << std::endl;
//	std:: cout << name << std::endl;
//	for (int i = 0; i < M; ++i) {
//		for(int j = 0; j < N; ++j){
//			printf("%.10f    ", a[i*lda + j]);
//		}
//		printf("\n");
//	}
//}

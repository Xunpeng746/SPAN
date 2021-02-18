CC	:= g++
MKLROOT := /home/lxf/bin/intel/compilers_and_libraries_2019.0.117/linux/mkl
LIBS    := -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_rt -liomp5 -lpthread -lm -ldl 
INCLUDE := -I. -I$(MKLROOT)/include -I$(MKLROOT)/include/intel64
CFLAGS  := -std=c++11 -g -w -fopenmp

CXXFLAGS:= $(CFLAGS) 
main: config.o Util.o Model.o Logistic.o  LSvm.o easylogging++.o Optimizer.o  Lissa.o Slbfgs.o Svrg.o GD.o NewSamp.o Sbbfgs.o Lant.o  Param.hpp Records.hpp Factory.o  TestUtil.o  MemFactory.hpp  getopt.hpp main.cpp  Context.hpp
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS)  -o  main config.o Util.o Model.o Logistic.o LSvm.o easylogging++.o Optimizer.o  Lissa.o Slbfgs.o Svrg.o GD.o NewSamp.o Sbbfgs.o Lant.o  TestUtil.o Factory.o getopt.hpp main.cpp 

train: config.o Util.o Model.o Logistic.o  LSvm.o easylogging++.o Optimizer.o  Lissa.o Slbfgs.o Svrg.o GD.o NewSamp.o Sbbfgs.o Lant.o  Param.hpp Records.hpp Factory.o  TestUtil.o  MemFactory.hpp  getopt.hpp main.cpp 
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS)  -o  train config.o Util.o Model.o Logistic.o LSvm.o easylogging++.o Optimizer.o  Lissa.o Slbfgs.o Svrg.o GD.o NewSamp.o Sbbfgs.o Lant.o  TestUtil.o Factory.o getopt.hpp main.cpp 

easylogging++.o: easylogging++.cc easylogging++.h
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS)  -c easylogging++.cc

config.o: config.cpp 
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS)  -c   config.cpp 

Util: config.h Util.cpp Util.h
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS)  -c config.h Util.cpp

Model.o: config.h Util.h Model.h Model.cpp config.h
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS) -c Model.cpp

Optimizer.o: config.h Model.h easylogging++.h Param.hpp Records.hpp Optimizer.cpp Optimizer.h
	g++  $(CXXFLAGS) $(INCLUDE) $(LIBS)  -c  Optimizer.cpp 

Logistic.o: Model.h Logistic.cpp Logistic.h
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS) -c Logistic.cpp

LSvm.o: Model.h LSvm.cpp LSvm.h
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS) -c LSvm.cpp

Lant.o: Optimizer.h Lant.cpp Lant.h
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS) -c  Lant.cpp 

NewSamp.o: Optimizer.h NewSamp.h NewSamp.cpp
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS) -c   NewSamp.cpp

Liisa.o: Optimizer.h Lissa.cpp Lissa.h
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS) -c Lissa.cpp 

Slbfgs.o: Optimizer.h Slbfgs.cpp Slbfgs.h Model.h
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS) -c  Slbfgs.cpp 

Sbbfgs.o: Optimizer.h Sbbfgs.cpp Sbbfgs.h Model.h
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS) -c   Sbbfgs.cpp 

GD.o: Optimizer.h GD.cpp GD.h
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS) -c GD.cpp 

svrg.o: Optimizer.h svrg.cpp svrg.h
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS) -c svrg.cpp 

Factory.o: config.h Model.h  Lissa.h Slbfgs.h NewSamp.h Svrg.h Lant.h Optimizer.h Logistic.h Param.hpp GD.h Sbbfgs.h  LSvm.h Factory.cpp Factory.h 
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS) -c Factory.cpp 

TestUtil.o: config.h Model.h Logistic.h LSvm.h Lant.h  Util.h TestUtil.h TestUtil.cpp
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS) -c TestUtil.cpp

Context.o: Factory.h config.h Util.h easylogging++.h Context.hpp
	g++  $(INCLUDE) $(LIBS) $(CXXFLAGS) -c  Context.hpp


clean:
	-rm *.o 
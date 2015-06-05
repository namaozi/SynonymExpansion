#all:a.out

CXX = g++
EIGEN_LOCATION=$$HOME/include/eigen
SQLITE3_LOCATION=$$HOME/include/sqlite
#OBJSをコンパイルするときのオプション...
CXXFLAGS = -Wall -g -std=gnu++0x
-lm -I -fopenmp -std=c++11 
CXXFLAGS += -O3
#CXXFLAGS += -pg #profiler...なんか付けないほうがいいらしい？
CXXFLAGS += -I$(EIGEN_LOCATION)
#これがうまくいかない?
#CXXFLAGS += -I$(SQLITE3_LOCATION)
#LDFLAGS = -l sqlite3
#LDFLAGS = -ldl
CXXFLAGS += -DEIGEN_NO_DEBUG
#もしかしたらいつか消したほうがいいかも...
CXXFLAGS += -DEIGEN_DONT_PARALLELIZE
OBJS = Main.o MyDictionary.o HelpTrain.o Word.o Evaluation.o
#sqlite3.o

a.out:	$(OBJS)
		$(CXX) $(OBJS) -o phrase -fopenmp 

clean:
	rm -f *~ *.o phrase

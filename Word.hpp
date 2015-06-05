/*
 * ヘッダファイルに記述すること
 * クラスの定義(変数)
 * インライン関数の定義
 * typedef宣言
 * グローバル関数の宣言
 * enum定義
 * 前方参照
 * 定数?
 */
#ifndef _MYD
#define _MYD

#include <stdio.h>
#include <Eigen/Core>
#include <map>
#include <unordered_map>
#include <string>

#define MAX_SIZE  3000
#define MAX_LEN  200
#define CBOW 0
#define LRBOW 1
#define WOLR 2
#define CWOLR 3

//行列表示用
#define PRINT_MAT(x) cout << #x << ":\n"<< x << endl;

//doubleかfloatかを選べる
//typedef float decimal;
typedef double decimal;

//doubleかfloatかを選べる
//typedef Eigen::MatrixXf MyMatrix;
typedef Eigen::MatrixXd MyMatrix;

/*
 * externはすべてここに書いてしまおう
 */
extern int PROJECT_ID;
extern int DIM;
extern int THRE;
extern int WINDOW_SIZE;
extern int SAMPLE;
extern int LAYER_SIZE;
extern int VOCAB_SIZE;
extern int TOTAL_LINE;
extern int REPEAT;
extern int EVAL;
extern int FLAG_DEFAULT;
extern int FLAG_DETAIL;
extern int FLAG_DESCENT;
extern int FLAG_INPUT;
extern int FLAG_OUTPUT;
extern int FLAG_APP;
extern long TOTAL_WORD;
extern long TEXT_SIZE;
extern decimal EPSILON;
extern decimal LAMBDA;

extern time_t timer;

extern std::vector<int> text_indice;
extern std::vector<int> text_test;
extern std::vector<int> text_training;
extern std::vector<decimal> vec_accuracy;
extern std::vector<decimal> vec_like;
extern std::vector<decimal> vec_like2;
extern std::unordered_map<int, decimal> index_to_prob;
extern std::unordered_map<int, std::string> index_to_word;
extern MyMatrix X, W, UnKnown_X, UnKnown_W;

class Word{
 public:
    int freq;
    int index;
};

class MyDictionary {

    //private
    std::unordered_map<std::string, Word> dic; //インスタンスはクラスが持っている

 public:
    //MyDictionary(){}; //コンストラクタ？
    void make_dictionary(std::string &filename, std::string &file_vocab, int N); //後々はn-gram実装したい
    //std::unordered_map<std::string, int> get_dic_freq();
    std::unordered_map<std::string, Word> get_dic();    
    //std::vector<std::string> split(const std::string &str, const std::string &delim);

};
//セミコロン忘れないようにね...

//クラスの外
std::vector<std::string> split(const std::string &str, const std::string &delim);

#endif //_MYD

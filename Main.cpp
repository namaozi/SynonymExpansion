/*
 * Contextを語順で区別したver.
 * 主な違い：H,Wの次元がWINDOW_SIZE * 2倍
 */
#include <Eigen/Core>
//#include "sqlite3.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <typeinfo>
#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include "Word.hpp"
#include "HelpTrain.hpp"
#include "Evaluation.hpp"


using namespace Eigen;
using namespace std;

//やっぱ実体確保はここで必要っぽい...

//パラメータ
int PROJECT_ID, DIM, THRE, WINDOW_SIZE, SAMPLE, REPEAT, EVAL;
decimal EPSILON, LAMBDA;
//フラグ集
int FLAG_DEFAULT, FLAG_DETAIL, FLAG_INPUT, FLAG_OUTPUT, FLAG_APP, FLAG_DESCENT;
//定数
int LAYER_SIZE, VOCAB_SIZE, TOTAL_LINE;
long TOTAL_WORD, TEXT_SIZE;
//行列
MyMatrix X, W, UnKnown_X, UnKnown_W;
//テキスト
vector<int> text_indice, text_test, text_training;
//結果保存用
vector<decimal> vec_accuracy, vec_like, vec_like2;
//変換辞書
unordered_map<int, string> index_to_word;
unordered_map<int, decimal> index_to_prob;

int main(int argc, char* argv[]){
    string filename, file_corpus, file_test, file_training, file_X, file_W, file_vocab, file_vec; //宣言
    string file_destination = "/home/ariga/Corpora/";
    string file_header = "/data/local/ariga/Vectors/";
    string data_type;
    FLAG_DEFAULT = 0;
    //適宜増やす
    if(argc >= 9){
        PROJECT_ID = atoi(argv[1]);
        DIM = atoi(argv[2]);
        THRE = atoi(argv[3]);
        WINDOW_SIZE = atoi(argv[4]);
        SAMPLE = atoi(argv[5]);
        EPSILON = atof(argv[6]);
        REPEAT = atoi(argv[7]);
        EVAL = atoi(argv[8]);
        if(argc >= 10){ //引数多い時
            FLAG_DESCENT = atoi(argv[9]);
            if(argc >= 11 && FLAG_DESCENT){ //学習率の下げ方
                LAMBDA = atof(argv[10]);
            }
            //FLAG_APP = 1;
        }
        if(DIM == 0 || THRE == 0 || WINDOW_SIZE == 0 ||EPSILON == 0 || SAMPLE == 0 ){
            cout << "please input integer (or decimal)."<< endl;
            return 1;
        }
    }else{//引数少ない時
        if(argc == 2 || argc == 4){
        string ddd = argv[1];        
            if(ddd == "d"){
                if(argc == 4){ //IO指定できる設定
                    FLAG_INPUT = atoi(argv[2]);
                    FLAG_OUTPUT = atoi(argv[3]);
                    if(FLAG_INPUT){
                        FLAG_APP = 1;
                        cout << "input X vector name:";
                        cin >> file_X;
                        cout << "input W vector name:";
                        cin >> file_W;
                        ifstream tmp_ifs_X, tmp_ifs_W;                            
                        tmp_ifs_X.open(file_header + file_X, ios::binary | ios::in);                            
                        tmp_ifs_W.open(file_header + file_W, ios::binary | ios::in);
                        if(!tmp_ifs_X){
                            cout<<"can't open the file."<<endl;
                            return 1;
                        }
                        //例を探すためのモード                            
                    }
                }
                /*
                 * デフォルトセッティング
                 */
                cout << "DEFAULT SETTING" << endl;
                PROJECT_ID = 2; //変えてね
                DIM = 100;
                THRE = 20;
                WINDOW_SIZE = 5;
                SAMPLE = 5;
                EPSILON = 0.025;
                REPEAT = 5;
                EVAL = 300;
                FLAG_DEFAULT = 1;
                FLAG_DESCENT = 0; //0:線形 1:アレ 2:定数
                if(FLAG_DESCENT) LAMBDA = 0.0000001; //10e-7
                //!!!!
                FLAG_DETAIL = 1; 
                //!!!!

                //file_corpus = "wiki.small";
                //file_corpus = "wiki.split.1";
                //file_corpus = "wiki.split.1-10";
                //file_corpus = "wiki.split.1-100";
                /*
                 * フレーズ!!!
                 */
                //file_corpus = "wiki.phrase2.small";                
                //file_corpus = "wiki.phrase2.1";
                //file_corpus = "wiki.phrase2.10";
                file_corpus = "wiki.phrase2.100";
                
            }else{
                cout << "please input 'd' or parameters." << endl;
                return 0;
            }
        }else{//デフォルト以外は弾かれる        
        cout << "please input parameters or type 'd' as default setting." << endl;
        cout << "1 : project ID(CBOW : 0, LRBOW : 1, WOLR : 2, CWOLR : 3)" << endl;
        cout << "2 : DIMENSION(int 100)" << endl;
        cout << "3 : THRESHOLD(int 10)" << endl;
        cout << "4 : WINDOW SIZE(int 5)" << endl;
        cout << "5 : number of NEGATIVE SAMPLING(int 5)" << endl;
        cout << "6 : training rate EPSILON(decimal 0.025)" << endl;
        cout << "7 : number of ITERATION(int 5)" << endl;
        cout << "8 : number of EVALUATION(int 500)" << endl;        
        cout << "9 : DESCENT FLAG(int, 0:Linear, 1:Reciprocal, 2:Constant)" << endl;
        cout << "*10:LAMBDA(decimal 0.000001)" << endl;
        return 0;
        }
    }

    /*
     * 学習ファイルを選択(デフォルト以外)
     */
    if(!FLAG_DEFAULT){
        //file_corpus = "wiki.small";
        //file_corpus = "wiki.split.1";
        //file_corpus = "wiki.split.1-10";
        //file_corpus = "wiki.split.1-100";        
        //file_corpus = "wiki.phrase2.1";
        //file_corpus = "wiki.phrase2.10";
        file_corpus = "wiki.phrase2.100";
        //!!!!
        FLAG_DETAIL = 1;
        //!!!!
    }
    /*
     * 評価ファイルを選択
     */
    //file_test = "wiki.split.301";
    file_training = "wiki.split.1";    
    //フレーズ
    file_test = "test.phrase2";

    //学習&評価ファイルをプリント
    filename = file_destination + file_corpus;
    file_test = file_destination + file_test;
    file_training = file_destination + file_training;
    cout << "training:" << filename << endl;
    cout << "test data:" << file_test << endl;
    //floatかdoubleか
    if(typeid(decimal) == typeid(double)) 
        data_type = "double";
    else 
        data_type = "float";
    cout << "data type:" << data_type << endl;
    //出入力設定(I/O設定時以外)
    if(argc != 4){
        if(FLAG_DEFAULT){ //デフォルトの時はベクトル保存してない
            FLAG_INPUT = FLAG_OUTPUT = 0;
        }else if(FLAG_APP){ //アプリモード時はベクトル読み込み
            FLAG_INPUT = 1;
            FLAG_OUTPUT = 0;
            //!!!
            FLAG_DETAIL = 2;
            //!!!
        }else{ //引数ありの場合は基本ベクトル保存
            FLAG_INPUT = 0;
            FLAG_OUTPUT = 1;
        }
    }
    //書き出しファイルの設定
    if(FLAG_OUTPUT || FLAG_INPUT){
        if(FLAG_OUTPUT) cout << "OUTPUT MODE" <<endl;
        if(FLAG_INPUT) cout << "INPUT MODE" <<endl;
        if(FLAG_APP){ //引数でファイル指定した時
            file_X = file_header + file_X;
            file_W = file_header + file_W;
            cout << "input X : " << file_X << endl;
            cout << "input W : " << file_W << endl;
        }else{
            file_X = (file_header + "X_");
            file_W = (file_header + "W_");
            //file_vocab = (file_header + "vocab_");

            time_t timer;
            struct tm *t_st;
            time(&timer);
            t_st = localtime(&timer);
            string tmp_time;
            tmp_time += to_string((long long int)t_st->tm_mon+1) + "_";
            tmp_time += to_string((long long int)t_st->tm_mday);

            string params;
            params += "M" + to_string((long long int)PROJECT_ID) + "_";
            params += "D" + to_string((long long int)DIM);
            //params += "E" + to_string((long decimal)EPSILON) + "_";
            //params += "I" + to_string((long long int)REPEAT);

            file_X += data_type + "_" + tmp_time + "_" + file_corpus + "_" +params;
            file_W += data_type + "_" + tmp_time + "_" + file_corpus + "_" +params;
            cout << file_X << endl;
        }        
    }
    MyDictionary dictionary;
    dictionary.make_dictionary(filename, file_vocab, 1); //ここでtext_indice作成
    unordered_map<string, Word> dic; //単語で索引、index,freqを参照可能

    //INPUTの時はファイルからdic作りたいね...

    dic = dictionary.get_dic(); 
    //評価用のデータ作成
    make_data_sample(dic, file_test, text_test); //テストデータ
    make_data_sample(dic, file_training, text_training); //学習データ

    cout << "vocab size:" << VOCAB_SIZE << endl;    
    cout << "total word:" << TOTAL_WORD << endl;        
    cout << "total line:" << TOTAL_LINE << endl;        

    //レイヤサイズ:中間層と重みベクトルの次元は何倍なのか 
    //CBOW : 1, LRBOW : 2, WOLR : 2 * WINDOW_SIZE      
    if(PROJECT_ID == CBOW){
        LAYER_SIZE = 1;
    }else if(PROJECT_ID == LRBOW){
        LAYER_SIZE = 2;
    }else if(PROJECT_ID == WOLR){
        LAYER_SIZE = 2 * WINDOW_SIZE;
    }else if(PROJECT_ID == CWOLR){
        LAYER_SIZE = 2 * WINDOW_SIZE + 1; //LRも入れる？
    }else{
        cout << "error : PROJECT_ID is invalid." <<endl;
        exit(1);
    }

    //ベクトル読み出し(file_X, file_Wから)
    if(FLAG_INPUT){ 
        ifstream ifs_X, ifs_W;
        ifs_X.open(file_X, ios::binary | ios::in);
        ifs_W.open(file_W, ios::binary | ios::in);
        if(!ifs_X){
            cout<<"can't open the file."<<endl;
            return 1;
        }
        //PROJECT_ID, DIMは一致している必要がある
        int tmp_id = 0, tmp_dim = 0, tmp_vsize = 0, tmp_lsize = 0;
        ifs_X.read((char*) &tmp_id, sizeof(int));
        ifs_X.read((char*) &tmp_dim, sizeof(int));
        ifs_X.read((char*) &tmp_vsize, sizeof(int));
        ifs_X.read((char*) &tmp_lsize, sizeof(int));        
        //後半読む意味ない.
        int buffer[100];
        ifs_W.read((char*) &buffer, sizeof(int) * 4);

        cout << "project id:" << tmp_id << ", dim:" << tmp_dim << ",vsize:" << tmp_vsize << endl; 
        if(tmp_vsize != VOCAB_SIZE){ //VOCAB_SIZEだけは一致していて欲しい
            cout << "this vector is not appropriate.(vocab size is different)" << endl;
            exit(1);
        }
        //アツい代入
        PROJECT_ID = tmp_id;
        DIM = tmp_dim;
        LAYER_SIZE = tmp_lsize;

        cout << "now loading..." << endl;
        X = MyMatrix::Zero(DIM, VOCAB_SIZE);
        W = MyMatrix::Zero(DIM * LAYER_SIZE, VOCAB_SIZE); //Zero

        //辞書をベクトルから読んで作りたい.......
        //freqはevaluateでは使わないので適当でいいはず
        for(int j = 0; j < VOCAB_SIZE; ++j){ //列は同じ次元
            //まず単語は何文字なのかを書いてから単語書く、じゃないと読めない
            string tmp_word = index_to_word[j];
            int tmp_size = tmp_word.size();
            ifs_X.read((char*) &tmp_size, sizeof(int));
            ifs_W.read((char*) &tmp_size, sizeof(int));
            for(int moji = 0; moji < tmp_size; ++moji){
                char tmp_char = tmp_word[moji];
                ifs_X.read( &tmp_char, sizeof(char));
                ifs_W.read( &tmp_char, sizeof(char));
            }
            //cout << tmp_word <<endl;
        }        

        for(int i = 0; i < DIM*LAYER_SIZE; ++i){ //行の次元が異なる
            //cout << i << endl;
            if(i < DIM){               
                for(int j = 0; j < VOCAB_SIZE; ++j){ //列は同じ次元
                    //ifs_X.read(( char * ) &X(i,j), sizeof( decimal ) );
                    //ifs_W.read(( char * ) &W(i,j), sizeof( decimal ) );                    
                    decimal val_X = 0, val_W = 0;
                    ifs_X.read(( char * ) &val_X, sizeof( decimal ) );
                    ifs_W.read(( char * ) &val_W, sizeof( decimal ) );
                    X(i, j) = val_X;
                    W(i, j) = val_W;
                }                
            }else{
                for(int j = 0; j < VOCAB_SIZE; ++j){
                    decimal val_W2 = 0;
                    ifs_W.read(( char * ) &val_W2, sizeof( decimal ) );                                        
                    W(i, j) = val_W2;
                }                
            }            
        }
        cout << endl;
        //PRINT_MAT(X) PRINT_MAT(W);

        while(1){// app mode
            FLAG_DETAIL = 2;
            EVAL = 50;
            cout<<"quit->q, change text->t, change EVAL->e, search->s, continue->OTHER :";
            string mode;
            cin >> mode;
            if(mode == "q"){
                break;
            }else if(mode == "t"){ //テキスト変える
                cout << "file name :";
                cin >> file_test;
                text_test.clear();
                make_data_sample(dic, file_test, text_test);
            }else if(mode == "e"){//EVAL回数変える
                cout << "EVAL :";
                cin >> mode;            
                EVAL = atoi(mode.c_str());
            }else if(mode == "s"){ //単語探す
                evaluate(-1, dic, text_test); //フラグ制御するしかない

            }else if(mode == "c"){
                evaluate(0, dic, text_test);
            }
        }
    }else{ //INPUTじゃない時
        //exit(1);
        //行列実体定義
        X = MyMatrix::Random(DIM, VOCAB_SIZE) / DIM;
        W = MyMatrix::Random(DIM * LAYER_SIZE, VOCAB_SIZE) / DIM; //Zero
        //ここでいろいろ設定を書けばいちいち動かさなくて済む

        // for(int j = 3; j >= 0; --j){
        //     PROJECT_ID = j;
        //     for(int i = 5; i <= 8; ++i){
        //         EPSILON = 0.005 * i;
        train(dic, file_X, file_W);
        //        cout << endl;
        //     }
        // }

    }    

    //とりあえず評価中は以下いらない
    /*
    string mode;
    while(1){
        make_ranking(dic);
        //cout<<"continue?(y/n)";
        //cin>>mode;
        //if(mode == "n") break;
    }
    */


    return 0;
    
}

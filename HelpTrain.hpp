#ifndef _HTR
#define _HTR

#include "Word.hpp"
#include <Eigen/Core>


//インライン関数はヘッダファイル内に実装
inline unsigned long xor_rand(){
    //time(&timer);
    static unsigned long x = 123456789;
    static unsigned long y = 362436069;
    static unsigned long z = 521288629;
    static unsigned long w = 88675123;
    //localtime(&timer)->tm_sec * 3203 * 653; 
    unsigned long t;
    t = x ^ (x << 11);
    x = y; y = z; z = w;
    return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)); 
};

//sigmoid[0,1]
inline decimal sigmoid(decimal a){
    return 1.0 / (1 + exp(-a));
};

//true:skip
inline bool judge_skip(decimal p, decimal r){
    if(p > 0.0){
        if(p < r){ //確率を上回った時は計算する
            return false; //0
        }
        else{
            return true; //1
        }
    }else{
        return false;
    }    
};

//H,Wの単語ベクトルに対応する開始位置を返す
inline int decide_position(int id_h, int id_center){
    if(PROJECT_ID == CBOW){
        return 0;
    }else if(PROJECT_ID == LRBOW){
        return (id_h < id_center) ? 0 : 1;
    }else if(PROJECT_ID == WOLR){
        return (id_h < id_center) ? (id_h + WINDOW_SIZE - id_center) : (id_h + WINDOW_SIZE - id_center - 1);
    }else{ //CWOLRも基本はWORLと同じ
        return (id_h < id_center) ? (id_h + WINDOW_SIZE - id_center) : (id_h + WINDOW_SIZE - id_center - 1);
    }
};

///////////////

decimal calc_objective(int flg, MyMatrix &H, int index_col);
void train(std::unordered_map<std::string, Word>&dic, std::string &file_X, std::string &file_W);


#endif //_HTR

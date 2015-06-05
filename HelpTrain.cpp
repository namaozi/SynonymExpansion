/*
 * 学習に関連する関数をまとめたファイル
 */
#include <Eigen/Core>
#include <math.h>
#include <stdlib.h>
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

using namespace std;
using namespace Eigen;

//1から中間層を作ってJの値を求める?
decimal calc_objective(int flg, MyMatrix &H, int index_col){

    if(flg){ //W
        return log(sigmoid(H.row(0) * W.col(index_col))) ;

    }else{ //X
        
        return log(sigmoid(H.row(0) * W.col(index_col))) ;

    }
}

//PROJECT_IDごとに初期化、更新のところを変える
//先にすすめてたやつ
void train(unordered_map<string, Word> &dic, string &file_X, string &file_W){
    //MyMatrix UnKnown_X = MyMatrix::Random(DIM,1) / DIM; //特別扱い
    //MyMatrix UnKnown_W = MyMatrix::Random(DIM * LAYER_SIZE, 1) / DIM;
    
    /* 学習率の下げ方のフラグ
     * 0 : 今までどおり(線形)
     * 1 : 割り算で下げるやつ
     * 2 : 定数
     */
    const decimal E0 = EPSILON;

    if(PROJECT_ID==0) cout<<"CBOW\n"; else if(PROJECT_ID==1) cout<<"LRBOW\n"; else if(PROJECT_ID==2) cout<<"WOLR\n"; else if(PROJECT_ID==3) cout<<"CWOLR\n"; else{cout<<"error:PROJECT ID is invalid.\n"; exit(1);}
    
    cout << "DIM = "<< DIM;
    cout << ", THRE = "<< THRE;
    cout << ", WINDOW_SIZE = "<< WINDOW_SIZE;
    cout << ", SAMPLE = " << SAMPLE;
    if(FLAG_DESCENT == 1) cout << ", E0 = " << E0;            
    else cout << ", EPSILON = " << EPSILON;        
    cout << ", ITERATION = " << REPEAT;            
    cout << ", EVAL = " << EVAL;            
    cout << ", DESCENT = " << FLAG_DESCENT;            
    if(FLAG_DESCENT == 1) cout << ", LAMBDA = " << LAMBDA << endl;            
    else cout << endl;

    long count_word = 0; //何文字目？
    long count_total_word = 1; //
    int progress_base = 4;
    int denom = TOTAL_WORD / progress_base;
    int progress = 0;
    decimal percent = 100.0 / progress_base; //percent%進むごとに表示
    decimal eps = EPSILON / (TOTAL_WORD * REPEAT);
    int ok_count = 0, ng_count = 0, total_ok_count = 0, total_ng_count = 0;
    decimal accuracy = 0, total_accuracy = 0;
    
    //複数回回せるように
    for(int count_repeat = 0; count_repeat < REPEAT; ++count_repeat){
        //極大値で終わりにする(いいベクトル作るときの設定)
        /*if(count_repeat >= 5){
            break;
            }*/
        count_word = 1;        
        if(FLAG_DETAIL){
            printf("%3d ",count_repeat + 1);
            cout << "iteration" << endl;
            cout << "count word:" << count_total_word << ", EPSILON=" << EPSILON << endl;
        }

        //TOTAL_WORD回窓かければ十分じゃない？    
        while(count_word < TOTAL_WORD){            

            //id(何番目)かは、本文の長さで割れば良い
            long id_center = xor_rand() % TEXT_SIZE;
            if(id_center == 0) continue; //なんか🙅
            //これがターゲットになる.text_indiceにidでアクセス
            int index_center = text_indice[id_center];
            string word_center = index_to_word[index_center];
            //cout << id_center << " " << index_center << " " << word_center << endl;
            //改行は単語カウントする前に判断
            if(index_center == -2){
                continue;
            }        
            ++count_word; //skipしたものもカウントするのでここでカウント
            ++count_total_word;

            //学習率変更
            if(FLAG_DESCENT == 0){ //線形
                if(EPSILON > 0.005)
                    EPSILON -= eps;                 
            }else if(FLAG_DESCENT == 1){ //割って下げるやつ,iterationを超えたカウンタをかけないとだめ...
                EPSILON = E0 / (1 + E0 * LAMBDA * count_total_word);                
                //cout << "EPSILON : " << EPSILON << endl;
            }//else : 定数(下げない)
            
            if(count_word % denom == 0){ //途中経過処理
                if(count_word == 0) continue;
                ++progress;
                accuracy = ok_count * 100.0 / (ok_count + ng_count);
                total_ok_count += ok_count;
                total_ng_count += ng_count;
                if(FLAG_DETAIL){
                    printf("in train word:%lu, progress:%3.1f%%, accuracy:%3.2f%%\n", count_word, progress * percent, accuracy);
                    //フラグ立ってれば頻繁に評価する
                    vec_accuracy.push_back(accuracy); //結果保存                    
                    evaluate(0, dic, text_test); 
                    //evaluate(1, dic, text_training);
                }
                accuracy = 0;
                ok_count = 0;
                ng_count = 0;                
            }
        
            //UNKNOWN
            if(index_center == -1){
                continue;
            }        
            long lower = id_center - WINDOW_SIZE; //下界
            long upper = id_center + WINDOW_SIZE; //上界
            //はみ出したらskip.
            if(upper >= TEXT_SIZE || lower <= 0){ 
                //cout << id_center << " " << index_center << " " << word_center << " ";
                //cout << "skipped." << endl;
                continue;
            }
            
            long aaa = 0;
            //改行を考慮して窓確定
            for(int k = 0; k < WINDOW_SIZE; ++k){
                aaa = WINDOW_SIZE - k;
                if(text_indice[id_center - aaa] == -2){ 
                    lower = id_center - aaa + 1;
                }
                if(text_indice[id_center + aaa] == -2){ 
                    upper = id_center + aaa - 1;
                }
            }
            if(lower == upper){ //1単語だけの行は殺す.
                continue;                    
            }
            //cout << "[" << lower << ", " << upper << "] " << id_center << endl;

            //skipするかどうか
            decimal r = 0;
            r = (xor_rand() % 10001) / 10000.0; 
            //subsampling確率を格納したindex_to_probに問い合わせる
            if(judge_skip( index_to_prob[index_center], r)){
                continue;
            }

            //毎回隠れ層初期化
            MyMatrix H = MyMatrix::Zero(1, DIM * LAYER_SIZE);
            //窓の下限〜上限を足してHを作る    
            //PROJECT_IDごとにベクトルのどこを使うかが異なる
            int start = 0, valid = 0; 
            for( long id_h = lower; id_h <= upper; ++id_h){   
                //cout << index_to_word[text_indice[id_h]] << " ";
                if(id_h == id_center){ //skip
                    continue;
                }                
                // プロジェクトごとに返り値が違います
                start = decide_position(id_h, id_center);
                //cout << start <<endl;
                int index_h = text_indice[id_h];
                if(index_h >= 0){  //辞書にある単語のとき   
                    ++valid;
                    H.block(0, start * DIM, 1, DIM) += X.col(index_h).transpose();
                    if(PROJECT_ID == CWOLR){ //一般化できなかったのでここで...
                        //さらに末尾にCBOWの層をつくる
                        //CBOWをWINDOW_SIZEで割ってみる...?
                        H.block(0, (LAYER_SIZE - 1) * DIM, 1, DIM) += X.col(index_h).transpose();// / (decimal)WINDOW_SIZE;
                    }
                }//UNKNOWN無視
            }
            //cout << endl;
            if(valid <= 0){ //UNKNOWNしかない 
                continue;
            }
            /*
             * GRADIENT CHECK
             * W_true
             */
            /*
            decimal delta = 0.0001;            
            decimal before, after, score_tmp, coeff_tmp, OBJ, diff;
            //forで回す
            for(int i=0; i<DIM*LAYER_SIZE; ++i){
                W.col(index_center)(i) += delta;
                before = calc_objective(0, H, index_center); //注意
                //cout <<"before"<< before << endl;

                W.col(index_center)(i) -= delta * 2;
                after = calc_objective(0, H, index_center);
                //cout <<"after"<< after << endl;

                W.col(index_center)(i) += delta;
                score_tmp = H.row(0) * W.col(index_center);                
                coeff_tmp = (1 - sigmoid(score_tmp));
                diff = H.row(0)(i) * coeff_tmp;
                OBJ =  (before - after) / (2.0 * delta);
                if(fabs(diff - OBJ) >= delta){
                    cout << "OBJ :" << OBJ << endl;
                    cout << "diff:" << diff << endl;
                    cout<<endl;
                }
            }
            */
            
            //正例のスコア計算,アップデート
            decimal coeff = 0, score_true = 0;
            //参照かもしれないけど、X更新はW固定なので代理変数置いておく
            MyMatrix W_center = W.col(index_center);
            //正例と負例のスコアを入れたvector.正答率調べるには必要.
            vector<decimal> vec_score;            

            //スコア計算はどのプロジェクトも同じ
            score_true = H.row(0) * W.col(index_center);                
            coeff =  EPSILON * (1 - sigmoid(score_true));
            start = 0; //初期化
            //スコアをvectorに突っ込む
            vec_score.push_back(score_true);
            for( long id_h = lower; id_h <= upper; ++id_h){//先にXcontextを更新
                if(id_h == id_center){
                    continue;
                }             
                start = decide_position(id_h, id_center); //プロジェクトごとの値                   
                int index_h = text_indice[id_h];
                if(index_h >= 0){ //X ok

                    X.col(index_h) += W_center.block(start * DIM, 0, DIM, 1) * coeff; //Wの一部のみ

                    if(PROJECT_ID == CWOLR){//末尾のCBOW成分は有効単語数で割る.
                        X.col(index_h) += W_center.block((LAYER_SIZE - 1) * DIM, 0, DIM, 1) * coeff;// / (decimal) WINDOW_SIZE;
                    }                
                }//UNKNOWNはとりあえず無視  
            }
            
            W.col(index_center) += H.row(0).transpose() * coeff; //Wを更新(共通処理)
            
            //負例をサンプリング
            //負例は頻度が高い単語ほどサンプリングされやすい(はず)
            for(int j = 0; j < SAMPLE; j++){
                //text_indiceからランダムにとってくる                
                int index_neg = text_indice[xor_rand() % TEXT_SIZE];
                //改行はじく！！！
                while(index_neg == index_center || index_neg < 0){
                    index_neg = text_indice[xor_rand() % TEXT_SIZE];
                }
                //cout << "NS " << j << " : "<< index_to_word[index_neg] << endl;
                decimal score_false = 0;
                //スコア計算は共通処理
                score_false = H.row(0) * W.col(index_neg);
                coeff = (-1) * EPSILON * sigmoid(score_false);
                start = 0; //初期化
                //スコアをvectorに突っ込む
                vec_score.push_back(score_false);
                for( long id_h = lower; id_h <= upper; ++id_h){//Xcontext
                    if(id_h == id_center){
                        continue;
                    }
                    start = decide_position(id_h, id_center); //プロジェクトごとの値                   
                    int index_h = text_indice[id_h];
                    if(index_h >= 0){ //X ok
                        X.col(index_h) += W.col(index_neg).block(start * DIM, 0, DIM, 1) * coeff; //Wの一部のみ
                        if(PROJECT_ID == CWOLR){
                            //末尾のCBOWの層を更新
                            X.col(index_h) += W.col(index_neg).block((LAYER_SIZE - 1) * DIM, 0, DIM, 1) * coeff;// / (decimal) WINDOW_SIZE;
                        }
                    }else{ // UnKnown
                        ;
                        //UnKnown_X += W_center.block(0,start,1,DIM) * coeff; 
                    }
                }
                W.col(index_neg) += H.row(0).transpose() * coeff;
            } 
            
            //学習中にaccuracy見る意味無いと感じるならいらない
            //max_element使った方が1.3倍くらい速くなるっぽい                        
            decimal score_first = *max_element(vec_score.begin(), vec_score.end());            

            if(score_first == score_true){ 
                ++ok_count; 
            }else{
                ++ng_count;
            }            
        }//while end

        //正答の平均順位(イテレーションごと)
        long total_rank = 0, total_score = 0;
        total_accuracy = (decimal)total_ok_count * 100.0 / (decimal)(total_ok_count + total_ng_count);
        if(FLAG_DETAIL){
            printf("TOTAL ACCCURACY : %5.2f\n", total_accuracy);
            cout<<endl;
        }else{
            //printf("%5.2f\n", total_accuracy);
        }

        if(!FLAG_DETAIL) vec_accuracy.push_back(total_accuracy); //結果保存
        total_accuracy = total_ok_count = total_ng_count = 0;
        total_rank = total_score = 0;
        
        //ここで学習&開発データで評価
        if(!FLAG_DETAIL){
            evaluate(0, dic, text_test); 
            //evaluate(1, dic, text_training);
        }
    }//iteration end

    //最終決算, 結果表示
    /*
    int number_eval;
    if(FLAG_DETAIL) number_eval = REPEAT * progress_base;
    else number_eval = REPEAT;
    */
    FLAG_DETAIL = 2;
    //最後にいろいろ表示しながら
    evaluate(0, dic, text_test); 
    
    cout << "accuracy" << endl;
    for(unsigned int i=0; i<vec_accuracy.size(); ++i){        
        cout << vec_accuracy[i] << endl;
    }
    cout << "likelihood(test)" << endl;
    for(unsigned int i=0; i<vec_like.size(); ++i){        
        cout << vec_like[i] << endl;
    }
    
    cout << "likelihood(training)" << endl;
    for(unsigned int i=0; i<vec_like2.size(); ++i){        
        cout << vec_like2[i] << endl;
    }

    //X,Wをファイル出力(バイナリできないのでしばらく諦め...)
    if(FLAG_OUTPUT && !FLAG_INPUT){
        ofstream ofs_X, ofs_W;
        ofs_X.open(file_X, ios::binary | ios::out);
        ofs_W.open(file_W, ios::binary | ios::out);

        //次元とかも保存しておく        
        ofs_X.write((char*) &PROJECT_ID, sizeof(int));
        ofs_X.write((char*) &DIM, sizeof(int));
        ofs_X.write((char*) &VOCAB_SIZE, sizeof(int));
        ofs_X.write((char*) &LAYER_SIZE, sizeof(int));        

        ofs_W.write((char*) &PROJECT_ID, sizeof(int));
        ofs_W.write((char*) &DIM, sizeof(int));
        ofs_W.write((char*) &VOCAB_SIZE, sizeof(int));
        ofs_W.write((char*) &LAYER_SIZE, sizeof(int));        

        //最初にindexに対応する単語を書く
        for(int j = 0; j < VOCAB_SIZE; ++j){ //列は同じ次元
            //まず単語は何文字なのかを書いてから単語書く、じゃないと読めない
            string tmp_word = index_to_word[j];
            int tmp_size = tmp_word.size();
            ofs_X.write((char*) &tmp_size, sizeof(int));
            ofs_W.write((char*) &tmp_size, sizeof(int));
            for(int moji = 0; moji < tmp_size; ++moji){
                char tmp_char = tmp_word[moji];
                ofs_X.write( &tmp_char, sizeof(char));
                ofs_W.write( &tmp_char, sizeof(char));
            }
        }        
        
        for(int i = 0; i < DIM*LAYER_SIZE; ++i){ //行の次元が異なる
            if(i < DIM){                               
                for(int j = 0; j < VOCAB_SIZE; ++j){ //列は同じ次元
                    decimal val_X = X(i,j); 
                    decimal val_W = W(i,j); 
                    //cout << (char *) &X(i, j) << " " ;
                    //cout << (char *) &val_X << endl;
                    ofs_X.write(( char * ) &val_X, sizeof( decimal ) );
                    ofs_W.write(( char * ) &val_W, sizeof( decimal ) );                    
                    //ofs_X.write(( char * ) &X(i,j), sizeof( decimal ) );
                    //ofs_W.write(( char * ) &W(i,j), sizeof( decimal ) );                    
                }
            }else{
                for(int j=0; j < VOCAB_SIZE; ++j){
                    decimal val_W2 = 0;
                    val_W2 = W(i, j);
                    ofs_W.write(( char * ) &val_W2, sizeof( decimal ) );                                        
                }                
            }            
        }                       
        //PRINT_MAT(X) PRINT_MAT(W);
        ofs_X.close();
        ofs_W.close();

    }

}

/*for(int k=0; k<SAMPLE+1; k++){
  if(!FLAG_DETAIL){//逆にフラグ折れてる時のみこれは表示
  printf("%9d ", piled_rank[k]);

  }
  total_rank += piled_rank[k];
  total_score += piled_rank[k] * (k+1);
  }       
  ave_rank = (decimal)total_score / (decimal)total_rank;
*/

/*
 * 評価に関連する関数をまとめたファイル
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

using namespace std;
using namespace Eigen;


//コサイン類似度測るやつ
void make_ranking(unordered_map<string, Word> &dic){

    string key;
    int index_target = 0;
    decimal score = 0;
    unordered_map<string, Word>::iterator it;
    map<decimal, string> result; //降順ソートするためにdecimalをキーに
    cout<< "input the word:";
    cin >> key; //単語入力
    it = dic.find(key);
    if(it != dic.end()){ 
        index_target = it->second.index;
        cout <<"\n"<< it->first << ": " << it->second.freq << "times appeared." << endl;
        for(unordered_map<string, Word>::iterator loop = dic.begin();
            loop != dic.end();
            ++loop){ //loopはdicの単語を初めから見ていく
            score = X.col(loop->second.index).transpose() * X.col(index_target);
            score /= (X.col(loop->second.index).norm() * X.col(index_target).norm()) ;
            result.insert(pair<decimal, string>(score, loop->first));
        }
        //スコアでソート
        int count = 1;
        for(map<decimal, string>::iterator ring = result.end(); ring != result.begin(); --ring){
            if(ring == result.end()) --ring; //929

            cout<< count << ":" << ring->second<< ":" << ring->first <<endl;
            ++count;
            if(count > 20) break;
        }    

    }else{ 
        cout << "can't find from the vocabulary." <<endl;        
    }
    return;
}

//尤度用のデータ作成
//学習データ、開発データ両者とも出来るように
void make_data_sample(unordered_map<string, Word> &dic, string &file_sample, vector<int> &sample_text){

    ifstream ifs_sample(file_sample);
    string line;
    int count_line = 0, count_word = 0, count = 0;
    if(ifs_sample.fail()){
        cout << "oooooooops!!!" << endl;
    }else{
        cout << "file :" + file_sample<< endl;
    }
    while(getline(ifs_sample, line)){
        count_line++;
        vector<string> words;
        words = split(line, " ");        
        count = words.size();
        //ここでもう変換させる
        for(int i=0; i<count; ++i){
            //cout << words[i] << " ";
            unordered_map<string, Word>::iterator it;
            it = dic.find(words[i]); 
            if(it != dic.end() ){
                sample_text.push_back(it->second.index);
            }else{//UNKNOWN
                sample_text.push_back(-1);
            }
            ++count_word;
        }
        sample_text.push_back(-2); //改行
        //cout << endl;
        //読みこむ行数の上限を決める
        if(count_line > 10000) break;
    }
    return;
}

//NSとは無関係に正答率を調べられるようにしたもの
//テストデータを引数にとる
//flgが-1の時は例を探すアプリモード
void evaluate(int flg, unordered_map<string, Word> &dic, vector<int> &sample_text){

    int count_word = 0, index_center = 0, index_aim = 0, total_rank = 0;
    long  id_vector = 0, id_center = 0;
    decimal likelihood = 0;
    string word_aim;
    vector<int> piled_rank(VOCAB_SIZE);
    vector<int> aim_indice; //検索ワードのid集合
    if(flg == -1){ //探索も一緒にやっちまえ        
        cout << "search word :";
        cin >> word_aim;        
        index_aim = dic[word_aim].index;
        //単語を検索してくる
        for(int i=0; i<(int)sample_text.size(); i++){
            if(sample_text[i] == index_aim){ //単語ヒットしたらベクトルに追加
                aim_indice.push_back(i); //idを格納,要注意
            }
        }        
        EVAL = aim_indice.size(); //注意
    }
    while(count_word < EVAL - 1){ //何回やる？
        //cout << count_word << "th, in" << EVAL <<endl; 
        //id_vectorはindiceの最初から何番目を見ているのか、を指すので
        //count_wordとは明確に使い分ける必要がある
        if(flg != -1){
            id_center = id_vector;
            index_center = sample_text[id_center];                
        }else{ //アプリモード
            id_center = aim_indice[id_vector]; //注意
            index_center = index_aim; //一定値
        }
        if(index_center < 0){
            ++id_vector;
            continue;
        }
        long lower = id_center - WINDOW_SIZE; //下界
        long upper = id_center + WINDOW_SIZE; //上界
        if(upper >= TOTAL_WORD || lower <= 0){
            ++id_vector;
            continue;        
        }
        //ここの数字超適当.適当にskipする
        //perplexity測る時ははずす???        
        if(flg != -1){
            decimal r = 0;
            r = (xor_rand() % 10001) / 10000.0; 
            if(judge_skip( index_to_prob[index_center], r) ){
                ++id_vector;
                continue;        
            }
        }
        
        //窓確定()
        long aaa = 0;
        for(int k = 0; k < WINDOW_SIZE; ++k){
            aaa = WINDOW_SIZE - k;
            if(sample_text[id_center - aaa] == -2) lower = id_center - aaa + 1;            
            if(sample_text[id_center + aaa] == -2) upper = id_center + aaa - 1;            
        }
        //中間層
        MyMatrix H = MyMatrix::Zero(1, DIM * LAYER_SIZE);
        int start = 0, valid = 0;
        for(long id_h = lower; id_h <= upper; ++id_h){   
            if(id_h == id_center) continue;            
            start = decide_position(id_h, id_center);
            int index_h = sample_text[id_h];
            if(index_h >= 0){  //辞書にある単語のとき   
                ++valid;
                H.block(0, start * DIM, 1, DIM) += X.col(index_h).transpose();
                if(PROJECT_ID == CWOLR){                     
                    //CBOWが効きすぎるので窓長で割りたい
                    H.block(0, (LAYER_SIZE - 1) * DIM, 1, DIM) += X.col(index_h).transpose() / (decimal)WINDOW_SIZE;
                }
            }
        }                
        if(valid <= 2){ //文に2単語以下とかはキモいので排除
            ++id_vector;
            continue;
        }
        //UNKNOWN = upper - lower - valid
        if(upper - lower - valid >= 2){ //UnKnown2個以上ある文を評価する気にならない
            ++id_vector;
            continue;
        }
        //ゴーサインが出たらカウント!!
        ++count_word;

        decimal score_true = 0, exp_true = 0;
        score_true = H.row(0) * W.col(index_center);                
        exp_true = exp(score_true);
        start = 0; //初期化
        //EVAL回サンプリングは欄外にコピペしました

        decimal exp_sum = 0, prob_now = 1;        
        /*
         *
         */
        //フラグ立ってれば順位と候補単語、コンテクストやfragmentノルムなど様々な情報を表示
        if(FLAG_DETAIL >= 2){

            //VOCAB全てからランキング作る
            //まず全単語のスコアをvectorとmapに対応付け.        

            vector<decimal> vec_score;
            unordered_map<decimal, string> score_to_word; //key:スコア、value:単語string        
            unordered_map<string, Word>::iterator it;
            for(it = dic.begin(); it != dic.end(); it++){
                int index_sample = it->second.index;
                decimal score_sample = H.row(0) * W.col(index_sample);
                vec_score.push_back(score_sample);
                //key:スコア、value:単語string
                score_to_word[score_sample] = it->first;
                exp_sum += exp(score_sample);
            }

            prob_now = exp_true / exp_sum ;
            likelihood += log(prob_now);
            //確率とか表示(いらないとは思うけど)
            //cout << "sum:" << exp_sum << ", ";
            //cout << "prob_now:" << prob_now <<endl;
            
            //降順ソート,score_trueが一番大きければ正解 
            sort(vec_score.begin(), vec_score.end(), greater<decimal>());
            //このkが順位を示す         
            int target_rank = 0; 
            for(int k=0; k < VOCAB_SIZE; k++){
                if(vec_score[k] == score_true){
                    ++piled_rank[k];
                    target_rank = k;
                    break;
                }
            }
            total_rank += target_rank;        

            //このへんのコードはほとんどが表示用        
            //適当に数字を設定。上位~%だけ表示
            if(target_rank < (VOCAB_SIZE / 100)){   
                //context表示(共通処理)
                //cout << endl;
                cout << "                 "; //前文
                for(long i = lower - 5; i <= lower; ++i){
                    int index_now = sample_text[i];
                    cout << index_to_word[index_now];
                    cout << "  ";
                }
                cout << endl;
                cout << "                 "; //表示調整用空白15
                for(long i = lower; i <= upper; ++i){
                    int index_now = sample_text[i];
                    if(index_now == index_center) cout << "[";
                    cout << index_to_word[index_now];
                    if(index_now == index_center) cout << "]";
                    cout << "  ";
                }
                cout << endl;
                cout << "                 "; //後文
                for(long i = upper; i <= upper + 5; ++i){
                    int index_now = sample_text[i];
                    cout << index_to_word[index_now];
                    cout << "  ";
                }
                cout << endl;
            
                for(int i=0; i<=1; ++i){ //2回だけ！
                    int relative_pos = 0;            
                    decimal inpro = 0;
                    //初回：1位に来た単語、2回目：中心の単語
                    string word_target;
                    if(i == 0){
                        word_target = score_to_word[vec_score[0]];
                    }else{
                        word_target = index_to_word[index_center];
                    }
                    int index_target = dic[word_target].index;
                    if(FLAG_DETAIL)
                        printf("%-15s ", word_target.c_str());
                    for(long l = lower; l <= upper; ++l){   
                        if(l == id_center){ //center skip
                            if(FLAG_DETAIL) cout << "CENTER ";
                            continue;
                        }
                        if(sample_text[l] < 0){
                            if(FLAG_DETAIL) cout << "XXX ";
                            continue;
                        }
                        relative_pos = decide_position(l, id_center);
                        MyMatrix H_tmp = H.block(0, relative_pos * DIM, 1, DIM);
                        MyMatrix W_tmp = W.block(relative_pos * DIM, index_target, DIM, 1);
                        inpro = H_tmp.row(0) * W_tmp.col(0);
                        //norm_dist[relative_pos] = inpro;
                        if(inpro > 0.0) cout<<" ";//表示調整用   
                        printf("%2.3f ",inpro);
                    }                
                    if(PROJECT_ID == CWOLR){ //CBOW部分を表示
                        MyMatrix H_tmp = H.block(0, (LAYER_SIZE - 1) * DIM, 1, DIM);
                        MyMatrix W_tmp = W.block((LAYER_SIZE - 1) * DIM, index_target, DIM, 1);
                        inpro = H_tmp.row(0) * W_tmp.col(0);
                        printf("CBOW:%2.3f ",inpro);
                    }
                    cout << endl;
                    if(vec_score[0] == score_true){ //1位
                        cout << "1st!!!"<<endl;
                        break;
                    }else if(i != 0){
                        cout << count_word << " " << target_rank + 1 << " : " << index_to_word[index_center] << " , " << vec_score[target_rank] << " (1st "<< score_to_word[vec_score[0]] << " : " << vec_score[0] << ")" << endl;                                        
                    }
                }//2回のfor end
                for(int i=0; i<5; ++i){//上位単語の表示
                    decimal tmp_score = vec_score[i];
                    string tmp_word = score_to_word[tmp_score];
                    printf("%2d %-15s %2.3f\n", i+1, tmp_word.c_str(), tmp_score);
                }            
                cout<<endl;

            }//if 上位 end
            else{ //下位はとりあえず何もしない
                ;
            }
        }//if DETAIL end
        //DETAILでない場合は、ひたすらに尤度のみを求める。
        else{
            //イテレーションなくても行列演算で確率計算できる！
            //1*dim, dim*vocab = 1*vocab
            MyMatrix S = H.row(0) * W;
            for(int j=0; j<VOCAB_SIZE; ++j){                
                exp_sum += exp( S(0, j) );
            }
            prob_now = exp_true / exp_sum;
            likelihood += log(prob_now);
        }
        ++id_vector;//////最後にインクリメント,超注意
    }//while end

    if(FLAG_DETAIL >= 2){
        decimal accuracy = (decimal) piled_rank[0] / (decimal) EVAL;
        decimal ave_rank = (decimal) total_rank / (decimal) EVAL;
        cout << "accuracy : " << accuracy << ", ";
        cout << "ave_rank : " << ave_rank << endl;
        //decimal perplexity = pow(likelihood , (decimal)-1 / (decimal)EVAL);
    }
    if(flg == 0){
        cout << "test    " ; 
    }else{
        cout << "training" ;
    }
    cout << "likelihood : " << likelihood << ", "; //データ数で割った方がいいのでは？
    cout << "ave_likely : " << likelihood / (decimal)EVAL << endl;
    if(flg == 0){
        vec_like.push_back(likelihood / (decimal)EVAL); //結果保存
    }else{
        vec_like2.push_back(likelihood / (decimal)EVAL); //結果保存
    }
    //piled_rank分布表示
    if(FLAG_DETAIL >= 2){
        for(int k=0; k<10; k++){    
            cout << k+1 << ":" << piled_rank[k] << " ";
        }
        cout << endl;
    }
}


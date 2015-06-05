/*
 * 引数によってn-gramの辞書を作成する
 * 
 */
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <fstream>
#include "Word.hpp"

using namespace std;

//空コンストラクタ
/*MyDictionary::MyDictionary()
{
}*/


//ngram 
//1117のやつからとってきたものからなおそうとしている
//改行もつけるよ
void MyDictionary::make_dictionary(string &filename, string &file_vocab, int N){ 
    
    ifstream ifs(filename);
    string line;
    int count_line = 0;
    int count = 0; //カウンタ設定(行の単語数)
    TOTAL_WORD = 0; //初期化
    TOTAL_LINE = 0;
    if(FLAG_DETAIL > 0) cout << "now making dictionary..." << endl;
    //1行読む
    while(getline(ifs, line)){
        count_line++;
        vector<string> words;
        words = split(line, " ");        
        count = words.size();
        
        //辞書に入れる処理
        for(int i=0;i<count;++i){
            ++TOTAL_WORD;
            unordered_map<string, Word>::iterator it;
            it = dic.find(words[i]);
            //マップが end では無い場合（つまりキーにヒットする値が存在した場合）
            //イテレータは->で指定できる(もしくは(*it).firstみたいにもできるはず)
            if( it != dic.end() ){
                it->second.freq += 1;
                //cout << "Word->freq:" << dic[ words[i] ].freq  << endl;
            }else{ //新単語を格納                
                Word tmp;
                tmp.freq = 1;       
                tmp.index = 0; //初期化
                dic.insert(pair<string, Word>(words[i], tmp));
            }            
        }
        //if(count > 100) break;      
    }

    int index_now = 0; //ゼロ始まり
    int freq_now = 0;
    //1.dicxに登録されているthreshold以下の頻度の単語を消して、単語にインデックスをつけていく
    //3.index_to_wordを作る
    //4.index_to_probを作る
    unordered_map<string, Word>::iterator it2 = dic.begin();
    decimal p = 0, q = 0;
    while(it2 != dic.end()){
        freq_now = it2->second.freq;
        if(freq_now >= THRE){
            //1.登録
            it2->second.index = index_now;

            //3.index_to_word作成
            index_to_word.insert(pair<int, string>(index_now, it2->first));
            
            //4.確率計算の後index_to_prob作成(pを格納)
            q = (decimal)(it2->second.freq) / TOTAL_WORD;
            p = 1 - sqrt(0.00001 / q); //10e^-5くらいが基準
            index_to_prob.insert(pair<int, decimal>(index_now, p));
            
            //cout << it2->first <<":"<< it2->second.freq<<", "<<it2->second.index<<", "<<endl;
            //インクリメント忘れずに.
            ++index_now;
        }else{
            dic.erase(it2); //要素削除
            //++it2; 
            //mapではここでもインクリメント必要,unorderedだと要らないらしい
        }        
        ++it2; //絶対必要
    }
    VOCAB_SIZE = 0; //初期化は必要っぽい？
    VOCAB_SIZE = index_now;
    //最後UnKnownと改行入れておく
    index_to_word.insert(pair<int, string>(-1, "UnKnown"));    
    index_to_word.insert(pair<int, string>(-2, "/_/"));    

    //もう1周して本文をインデックスの形に変更してメモリ上に置く
    if(FLAG_DETAIL > 0) cout << "now converting text to index..." << endl;
    count_line = 0;
    ifstream ifs2(filename);
    while(getline(ifs2, line)){
        ++count_line;
        vector<string> words;
        words = split(line, " ");        
        count = words.size();
        for(int i=0;i<count;++i){
            it2 = dic.find(words[i]);
            if(it2 != dic.end()){
                text_indice.push_back( it2->second.index );
            }else{ //UNKNOWN
                text_indice.push_back((int)-1 );
            }
        }
        //改行くわえる
        text_indice.push_back((int)-2 );
    }

    TOTAL_LINE = count_line;
    TEXT_SIZE = TOTAL_WORD + TOTAL_LINE;
}


unordered_map<string, Word> MyDictionary::get_dic(){
    return dic;
}



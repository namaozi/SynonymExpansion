#ifndef _EVL
#define _EVL



///////////////

void make_ranking(std::unordered_map<std::string, Word> &dic);
void make_data_sample(std::unordered_map<std::string, Word> &dic, std::string &file_sample, std::vector<int> &sample_text);
void evaluate(int flg, std::unordered_map<std::string, Word> &dic, std::vector<int> &sample_text);

#endif //EVL

#ifndef DICTIONARY_H
#define DICTIONARY_H

#include <algorithm>
#include <string>
#include <vector>

class Dictionary
{
 protected:
  void read_words_from_file(const char *path);

 public:
  const std::vector<char> alphabet_;
  uint32_t alphabet_size_;
  uint32_t max_word_len_ = 0;
  std::vector<std::string> strings_;
  std::vector<std::vector<int32_t>> words_;
  std::vector<std::vector<int32_t>> freqs_;
  int32_t max_counts_ = 0;
  int32_t **counts_;
  int32_t **starts_;
  int32_t ***positions_;
  int32_t ****labeled_freqs_;

  std::vector<std::string> letter_strings_;
  std::vector<std::vector<int32_t>> letter_words_;
  std::vector<std::vector<int32_t>> letter_freqs_;

  std::string blank_string_;
  std::vector<int32_t> blank_word_;
  std::vector<int32_t> blank_freq_;

  Dictionary(const char* path, const std::vector<char> &alphabet);
  ~Dictionary();

  bool exists(const std::vector<int32_t> &word) const;
  int32_t get_max_word_length() const { return max_word_len_; };
};

#endif

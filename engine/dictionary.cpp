#include "dictionary.h"

Dictionary::Dictionary(const char* path, const std::vector<char> &alphabet) : alphabet_{alphabet}
{
  alphabet_size_ = alphabet_.size();

  // Try to open the dictionary
  FILE* file = fopen(path, "r");
  if (file == NULL)
    exit(-1);

  // Go through each line in the file and if the line begins with at least two letters, add the letters
  size_t len = 0;
  uint32_t let, word_len;
  int letter_pos;
  char* line;
  while (getline(&line, &len, file) != -1)
    {
      std::vector<int> word;
      for (word_len = 0; line[word_len] > ' '; ++word_len)
	{
	  letter_pos = find(alphabet_.begin(), alphabet_.end(), line[word_len]) - alphabet_.begin();
	  if (letter_pos >= 0)
	    word.push_back(letter_pos);
	  else
	    fprintf(stderr, "Unknown letter [%i]\n", line[word_len]);
	}
      if (word_len < 2) continue;
      words_.push_back(word);
      if (word_len > max_word_len_)
	max_word_len_ = word_len;
    }

  // Initialize a counter of word lengths 
  counts_ = new int32_t*[max_word_len_ + 1];
  starts_ = new int32_t*[max_word_len_ + 1];
  int32_t **counter = new int32_t*[max_word_len_ + 1];
  for (len = 0; len <= max_word_len_; ++len)
    {
      counts_[len] = new int32_t[alphabet_size_]();
      starts_[len] = new int32_t[alphabet_size_]();
      counter[len] = new int32_t[alphabet_size_]();
    }

  // Sort words for easy searching then add their freqs
  std::sort(words_.begin(), words_.end());
  for (auto &word : words_)
    {
      std::vector<int> freq(alphabet_size_, 0);
      std::vector<char> string;
      for (auto &l : word)
	{
	  ++freq[l];
	  string.push_back(alphabet_[l]);
	}
      for (let = 0; let < alphabet_size_; ++let)
	{
	  counts_[word.size()][let] += (freq[let] > 0);
	  /* GPU 
	  h_freqs_[let] = freq[let];
	  */
	}
      freqs_.push_back(freq);
      strings_.push_back(std::string(string.begin(), string.end()));
    }
  
  // Allocate memory for optimized searching by word len and contained character .
  positions_ = new int**[max_word_len_ + 1];
  labeled_freqs_ = new int***[max_word_len_ + 1];
  for (len = 0; len <= max_word_len_; ++len)
    {
      positions_[len] = new int*[alphabet_size_];
      labeled_freqs_[len] = new int**[alphabet_size_];
      for (let = 0; let < alphabet_size_; ++let)
	{
	  positions_[len][let] = new int[counts_[len][let]];
	  labeled_freqs_[len][let] = new int*[counts_[len][let]];
	}
    }

  // Map our arrays
  for (unsigned int pos = 0; pos < words_.size(); ++pos)
    {
      len = words_[pos].size();
      for (let = 0; let < alphabet_size_; ++let)
	if (freqs_[pos][let] > 0)
	  {
	    positions_[len][let][counter[len][let]] = pos;
	    labeled_freqs_[len][let][counter[len][let]] = &freqs_[pos][0];
	    ++counter[len][let];
	  }
    }

  // Create single letter freqs and stuff
  for (let = 0; let < alphabet_size_; ++let)
    {
      letter_strings_.push_back(std::string(1, alphabet_[let]));
      letter_words_.push_back(std::vector<int32_t>(1, let));
      letter_freqs_.push_back(std::vector<int32_t>(alphabet_size_, 0));
      letter_freqs_.back()[let] = 1;
    }

  blank_freq_ = std::vector<int32_t>(alphabet_size_, 0);

  // Free counter
  for (len = 0; len <= max_word_len_; ++len)
    delete[] counter[len];
  delete counter;
}

Dictionary::~Dictionary()
{
  unsigned int let, len;
  for (len = 0; len <= max_word_len_; ++len)
    {
      for (let = 0; let < alphabet_size_; ++let)
	{
	  delete[] positions_[len][let];
	  delete[] labeled_freqs_[len][let];
	}
      delete[] positions_[len];
      delete[] labeled_freqs_[len];
    }
  delete[] positions_;
  delete[] labeled_freqs_;
  for (len = 0; len <= max_word_len_; ++len)
    delete[] counts_[len];
  delete[] counts_;
}

bool Dictionary::exists(const std::vector<int> &word) const
{  
  return std::binary_search (words_.begin(), words_.end(), word);
}


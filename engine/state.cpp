#include "state.h"

StateAction::StateAction(const StateAction &s) 
  : alphabet_size_(s.alphabet_size_), vector_size_(s.vector_size_)
{
  vector_ = new int32_t[vector_size_];
  for (uint32_t i = 0; i < vector_size_; ++i)
    vector_[i] = s.vector_[i];
}

StateAction::StateAction(uint32_t bunch_size, int32_t bunch_count, uint32_t alphabet_size, 
			 const int32_t* freq_hand, const int32_t* freq_growable, const int32_t* freq_wiltable,
			 const int32_t* age_hand)
  : alphabet_size_(alphabet_size)
{
  vector_size_ = 5 * (alphabet_size_ + 1);
  vector_ = new int32_t[vector_size_];
  
  int32_t pos = -1;

  vector_[++pos] = 0; // Peel
  vector_[++pos] = 0; // Dump
  vector_[++pos] = 0; // Grow
  vector_[++pos] = 0; // Wilt
  vector_[++pos] = bunch_count;

  for (uint32_t i = 0; i < alphabet_size_; ++i)
    {
      vector_[++pos] = 0; // Action
      vector_[++pos] = freq_hand[i];
      vector_[++pos] = freq_growable[i];
      vector_[++pos] = freq_wiltable[i];
      vector_[++pos] = age_hand[i];
    }
}

void StateAction::update_action(char action, const int *freq)
{
  vector_[0] = action == 'D';
  vector_[1] = action == 'G';
  vector_[2] = action == 'W';
  vector_[3] = action == 'P';
  for (uint32_t i = 0; i < alphabet_size_; ++i)
    vector_[(i + 1) * 5] = freq[i];
}

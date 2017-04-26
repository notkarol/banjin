#ifndef STATEACTION_H
#define STATEACTION_H

#include <iostream>
#include <vector>
#include <string>

class StateAction
{
 protected:
  uint32_t alphabet_size_;
  uint32_t vector_size_;
  int32_t *vector_;
  
 public:
  StateAction();
  StateAction(const StateAction &s);
  StateAction(uint32_t bunch_size, int32_t bunch_count, uint32_t alphabet_size, const int32_t* freq_hand, 
	      const int32_t* freq_growable, const int32_t* freq_wiltable, const int32_t* age_hand);

  void update_action(char action, const int *freq);

  int32_t operator[](int position) { return vector_[position]; }
  uint32_t size() { return vector_size_; }
};

#endif

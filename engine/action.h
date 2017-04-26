#ifndef ACTION_H
#define ACTION_H

#include <algorithm>
#include <iostream>
#include <vector>
#include <string>

class Action
{
 protected:
  const std::string *string_;
  const std::vector<int> *word_;
  const std::vector<int> *freq_;
  char action_;
  int x_ = 0;
  int y_ = 0;
  bool down_ = false;
  float value_ = 0.0;

 public:
  Action();
  Action(const Action &m);
  Action(const std::string *string, const std::vector<int> *word, const std::vector<int> *freq, 
	 char action, int x, int y, bool down, float value);

  // Operator overloading
  bool operator==(const Action& m);
  bool operator<(const Action& m);
  friend std::ostream& operator<<(std::ostream &output, const Action &m);
  
  // Setters
  void set_value(float value);
  void set_action(char action);

  // Getters
  const std::string* get_string() const { return string_; }
  const std::vector<int>* get_word() const { return word_; }
  const std::vector<int>* get_freq() const { return freq_; }
  char get_action() const { return action_; }
  int get_x() const { return x_; }
  int get_y() const { return y_; }
  bool get_down() const { return down_; }
  float get_value() const { return value_; }
  int get_size() const { return string_->size(); }
};

#endif

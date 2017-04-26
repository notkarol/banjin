#include "action.h"

Action::Action() 
{}

Action::Action(const Action &a) 
  : string_(a.string_), word_(a.word_), freq_(a.freq_), action_(a.action_), x_(a.x_), y_(a.y_), down_(a.down_), value_(a.value_)
{}

Action::Action(const std::string *string, const std::vector<int> *word, const std::vector<int> *freq, 
	       char action, int x, int y, bool down, float value)
  : string_(string), word_(word), freq_(freq), action_(action), x_(x), y_(y), down_(down), value_(value)
{}

bool Action::operator==(const Action &a)
{ 
  return action_ == a.action_ && x_ == a.x_ && y_ == a.y_ && down_ == a.down_; 
}

bool Action::operator<(const Action &a)
{ 
  return value_ < a.value_; 
}

std::ostream &operator<<(std::ostream &output, const Action &a)
{ 
  output << '(' << a.action_ << ',' << a.string_->c_str() << ',' << a.x_ << ',' << a.y_ << ',' << a.down_ << ',' << a.value_ <<  ')';
  return output;
}

void Action::set_value(float value) 
{ 
  value_ = value; 
}

void Action::set_action(char action) 
{ 
  action_ = action; 
}


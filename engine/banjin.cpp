#include "banjin.h"

Banjin::Banjin(int split_count, const std::vector<int> *bunch_init, const Dictionary *dictionary, uint64_t seed, int grid_x, int grid_y)
  : split_count_(split_count), dict_(dictionary), seed_(seed), grid_x_(grid_x), grid_y_(grid_y)
{
  rng_.seed(seed_);
  dist = std::uniform_real_distribution<double>(0.0, 1.0);
  
  // Prepare bunch
  freq_bunch_ = new int[dict_->alphabet_size_]();
  freq_hand_ = new int[dict_->alphabet_size_]();
  freq_grid_ = new int[dict_->alphabet_size_]();
  freq_growable_ = new int[dict_->alphabet_size_]();
  freq_wiltable_ = new int[dict_->alphabet_size_]();
  age_hand_ = new int[dict_->alphabet_size_]();

  // Initialize bunch frequencies
  std::vector<int> tile_letters;
  for (uint32_t i = 0; i < dict_->alphabet_size_; ++i)
    {
      freq_bunch_[i] = (*bunch_init)[i];
      num_tiles_ += freq_bunch_[i];
      for (int j = 0; j < freq_bunch_[i]; ++j)
	tile_letters.push_back(i);
      pos_in_hand_by_letter_.push_back(std::vector<int>());
    }
  std::shuffle(tile_letters.begin(), tile_letters.end(), rng_);

  // Initialize tiles
  tile_letter_ = new int[num_tiles_]();
  tile_x_ = new int[num_tiles_]();
  tile_y_ = new int[num_tiles_]();
  tile_in_bunch_ = new int[num_tiles_]();
  tile_in_hand_ = new int[num_tiles_]();
  tile_in_grid_ = new int[num_tiles_]();
  for (int i = 0; i < num_tiles_; ++i)
    {
      tile_letter_[i] = tile_letters[i];
      tile_in_bunch_[i] = 1;
      pos_in_bunch_.push_back(i);
    }

  // Initialize grid
  grid_letters_ = new int*[grid_y_];
  grid_neighbors_ = new int*[grid_y_];
  grid_positions_ = new int*[grid_y_];

  for (int i = 0; i < grid_y_; ++i)
    {
      grid_letters_[i] = new int[grid_x_];
      grid_neighbors_[i] = new int[grid_x_]();
      grid_positions_[i] = new int[grid_x_];
      for (int j = 0; j < grid_x_; ++j)
	{
	  grid_letters_[i][j] = -1;
	  grid_positions_[i][j] = -1;
	}
    }

  // Split here because it's not much of a choice anyway
  split();
}

// Free Memory
Banjin::~Banjin()
{
  for (int i = 0; i < grid_y_; ++i)
    {
      delete grid_letters_[i];
      delete grid_neighbors_[i];
      delete grid_positions_[i];
    }
  delete[] grid_letters_;
  delete[] grid_neighbors_;
  delete[] grid_positions_;

  delete[] tile_letter_;
  delete[] tile_x_;
  delete[] tile_y_;
  delete[] tile_in_bunch_;
  delete[] tile_in_hand_;
  delete[] tile_in_grid_;

  delete[] freq_bunch_;
  delete[] freq_hand_;
  delete[] freq_grid_;
  delete[] freq_growable_;
  delete[] freq_wiltable_;
  delete[] age_hand_;
}

void Banjin::split()
{
#ifdef VERBOSE
  std::cout << "SPLIT (" << split_count_ << ")\n";
#endif
  give(split_count_);
}

void Banjin::peel()
{ 
#ifdef VERBOSE
  std::cout << "PEEL [" << dict_->alphabet_[tile_letter_[pos_in_bunch_.back()]] << "]\n";
#endif
  give(1);
}


void Banjin::give(int count)
{
  int pos;
  while (--count >= 0)
    {  
      pos = pos_in_bunch_.back();

      // Update freqs
      --freq_bunch_[tile_letter_[pos]];
      ++freq_hand_[tile_letter_[pos]];

      // Update vectors
      pos_in_hand_.push_back(pos);
      pos_in_hand_by_letter_[tile_letter_[pos]].push_back(pos);
      pos_in_bunch_.pop_back();

      // Update tile 
      tile_in_hand_[pos] = 1;
      tile_in_bunch_[pos] = 0;
    }
}

void Banjin::take(int letter)
{
  int pos = pos_in_hand_by_letter_[letter].back();

  // Update Freqs
  --freq_hand_[letter];
  ++freq_bunch_[letter];

  // Update vectors
  pos_in_bunch_.push_back(pos);
  pos_in_hand_.erase(std::remove(pos_in_hand_.begin(), pos_in_hand_.end(), pos), pos_in_hand_.end()); 
  pos_in_hand_by_letter_[letter].pop_back();

  // Update tile
  tile_in_bunch_[pos] = 1;
  tile_in_hand_[pos] = 0;
 
  std::shuffle(pos_in_bunch_.begin(), pos_in_bunch_.end(), rng_);
}

void Banjin::grow(Action *match)
{
#ifdef VERBOSE
  std::cout << "Grow " << *match << "\n";
#endif
  ++num_grow_;
  int pos;
  int x = match->get_x();
  int y = match->get_y();
  for (auto &letter : *match->get_word())
    {
      if (grid_letters_[y][x] == -1)
	{
	// Get pos of tile from hand
	  pos = pos_in_hand_by_letter_[letter].back();

	  // Update freqs
	  --freq_hand_[letter];
	  ++freq_grid_[letter];
	  ++freq_wiltable_[letter];
	  ++freq_growable_[letter];

	  // Update vectors
	  pos_in_grid_.push_back(pos);
	  pos_in_hand_.erase(std::remove(pos_in_hand_.begin(), pos_in_hand_.end(), pos), pos_in_hand_.end());
	  pos_in_hand_by_letter_[letter].pop_back();

	  // Update tile 
	  tile_x_[pos] = x;
	  tile_y_[pos] = y;
	  tile_in_hand_[pos] = 0;
	  tile_in_grid_[pos] = 1;
	  
	  // Update Grid
	  grid_letters_[y][x] = letter;
	  grid_positions_[y][x] = pos;
	  ++grid_neighbors_[y_b(y)][x];
	  ++grid_neighbors_[y_a(y)][x];
	  ++grid_neighbors_[y][x_b(x)];
	  ++grid_neighbors_[y][x_a(x)];
	}
      else
	{
	  --freq_growable_[letter];
	  // If the attaching word only has one intersection, remove wiltable freq
	}
      x = (x + 1 - match->get_down()) % grid_x_;
      y = (y + match->get_down()) % grid_y_;
    }
  match->set_action('W');
  words_.push_back(*match);
}

void Banjin::wilt(Action *match)
{
#ifdef VERBOSE
  std::cout << "Wilt " << *match << "\n";
#endif
  ++num_wilt_;
  int pos;
  int x = match->get_x();
  int y = match->get_y();
  for (auto &letter : *match->get_word())
    {
      if (( match->get_down() && grid_letters_[y][x_b(x)] == -1 && grid_letters_[y][x_a(x)] == -1) ||
	  (!match->get_down() && grid_letters_[y_b(y)][x] == -1 && grid_letters_[y_a(y)][x] == -1))
	{
	  ++cost_;

	  // Get pos of tile from hand
	  pos = grid_positions_[y][x];

	  // Update vectors
	  pos_in_hand_.push_back(pos);
	  pos_in_hand_by_letter_[letter].push_back(pos);
	  pos_in_grid_.erase(std::remove(pos_in_grid_.begin(), pos_in_grid_.end(), pos), pos_in_grid_.end());

	  // Update freqs
	  --freq_grid_[letter];
	  ++freq_hand_[letter];
	  --freq_wiltable_[letter];
	  --freq_growable_[letter];
	  	  
	  // Update tile 
	  tile_x_[pos] = -1;
	  tile_y_[pos] = -1;
	  tile_in_grid_[pos] = 0;
	  tile_in_hand_[pos] = 1;
	  
	  // Update grid
	  grid_letters_[y][x] = -1;
	  grid_positions_[y][x] = -1;
	  --grid_neighbors_[y_b(y)][x];
	  --grid_neighbors_[y_a(y)][x];
	  --grid_neighbors_[y][x_b(x)];
	  --grid_neighbors_[y][x_a(x)];
	}
      else
	{
	  ++freq_growable_[letter];
	  // Check if there's no other intersection and then allow those letters to be wiltable
	}
      x = (x + 1 - match->get_down()) % grid_x_;
      y = (y + match->get_down()) % grid_y_;
    }
  words_.erase(std::remove(words_.begin(), words_.end(), *match), words_.end());
}

void Banjin::dump(Action *action)
{
  int letter = (*action->get_word())[0];
#ifdef VERBOSE
  std::cout << "Dump [" << dict_->alphabet_[letter] << "]\n";
#endif
  ++num_dump_;
  cost_ += 4;
  take(letter);
  give(3);
}

bool Banjin::is_done() const
{
  return pos_in_hand_.size() == 0 && pos_in_bunch_.size() == 0;
}

void Banjin::room(int x, int y, bool down, int *b, int *a)
{
  int tx, ty;

  ty = down * y_b(y) + (1 - down) * y;
  tx = (1 - down) * x_b(x) + down * x;
  *b = -1 + (grid_letters_[ty][tx] == -1) + (grid_letters_[ty][tx] == -1 && grid_neighbors_[ty][tx] <= 1);

  if (*b == 1)
    while (*b < (int) dict_->max_word_len_ && 
	   grid_neighbors_[down * y_b(ty - *b + 1) + (1 - down) * ty][down * tx + (1 - down) * x_b(tx - *b + 1)] == 0)
      ++*b;

  ty = down * y_a(y) + (1 - down) * y;
  tx = (1 - down) * x_a(x) + down * x;
  *a = -1 + (grid_letters_[ty][tx] == -1) + (grid_letters_[ty][tx] == -1 && grid_neighbors_[ty][tx] <= 1);

  if (*a == 1)
    while (*a < (int) dict_->max_word_len_ && 
	   grid_neighbors_[down * y_a(ty + *a - 1) + (1 - down) * ty][down * tx + (1 - down) * x_a(tx + *a - 1)] == 0)
      ++*a;
}

bool Banjin::is_end_tile(int pos)
{
  return grid_neighbors_[tile_y_[pos]][tile_x_[pos]] == 1;
}

bool Banjin::is_edge_tile(int pos)
{
  return (grid_neighbors_[tile_y_[pos]][tile_x_[pos]] == 2 &&
	  ((grid_letters_[y_b(tile_y_[pos])][tile_x_[pos]] == -1 &&
	    grid_letters_[y_a(tile_y_[pos])][tile_x_[pos]] == -1) ||
	   (grid_letters_[tile_y_[pos]][y_b(tile_x_[pos])] == -1 &&
	    grid_letters_[tile_y_[pos]][y_a(tile_x_[pos])] == -1)));
}

void Banjin::wiltable(std::vector<Action> *matches)
{
  int intersections, x, y;
  std::vector<int> letters;
  int age;

  for (auto &match : words_)
    {
      letters.clear();
      intersections = 0;
      x = match.get_x();
      y = match.get_y();
      while (grid_letters_[y][x] != -1)
	{
	  if ((!match.get_down() && (grid_letters_[y_b(y)][x] != -1 || 
						    grid_letters_[y_a(y)][x] != -1)) ||
			     ( match.get_down() && (grid_letters_[y][x_b(x)] != -1 || 
						    grid_letters_[y][x_a(x)] != -1)))
	    {
	      ++intersections;
	      letters.push_back(grid_letters_[y][x]);
	    } 
	  x = (x + 1 - match.get_down()) % grid_x_;
	  y = (y + match.get_down()) % grid_y_;
	}
      if (intersections <= 1)
	{
	  matches->push_back(match);

	  age = letters.size();
	  for (auto &l : letters)
	    {
	      ++freq_wiltable_[l];
	      age += std::sqrt(max_age_ - age_hand_[l]);
	    }
	  matches->back().set_value(age);
	}
    }
}

void Banjin::growable(std::vector<Action> *matches, int pos)
{
  int x = tile_x_[pos];
  int y = tile_y_[pos];
  int b = 0;
  int a = 0;
  if (grid_neighbors_[y][x] <= 2)
    for (auto &d : {true, false})
      {
	room(x, y, d, &b, &a);
	//printf("Found room (%i,%i) at (%i,%i) %i [%c]\n", b, a, x, y, d, dict_->alphabet_[tile_letter_[pos]]);
	if (a >= 0 && b >= 0 && (a > 0 || b > 0))
	  {
	    ++freq_growable_[tile_letter_[pos]];
	    growable(matches, x, y, d, b, a);
	  }
      }
}


void Banjin::growable(std::vector<Action> *matches, int x, int y, bool down, int b, int a)
{
  int max_len, letter;

  max_len = std::min((int) pos_in_hand_.size() + 1, std::max(b, a));
  if (grid_letters_[y][x] != -1)
    {
      letter = grid_letters_[y][x];
      ++freq_hand_[letter];
    }
  else 
    letter = tile_letter_[pos_in_hand_[0]]; 

  int word_pos, size, len, age, i;
  uint32_t let, pos;
  for (len = max_len; len >= 2; --len)
    {
      for (pos = 0; pos < (uint32_t) dict_->counts_[len][letter]; ++pos)
	{
	  for (let = 0; let < dict_->alphabet_size_; ++let)
	    if (dict_->labeled_freqs_[len][letter][pos][let] > freq_hand_[let])
	      break;
	  if (let < dict_->alphabet_size_)
	    continue;
	  word_pos = dict_->positions_[len][letter][pos];
	  size = dict_->words_[word_pos].size() - 1;

	  if (dict_->words_[word_pos][0] == letter && a >= size)
	    {
	      age = size;
	      for (i = 1; i <= size; ++i)
		age += age_hand_[dict_->words_[word_pos][i]] * 2;
	      matches->push_back(Action(&dict_->strings_[word_pos], &dict_->words_[word_pos], 
					&dict_->freqs_[word_pos], 'G', x, y, down, age));
	    }
	  if (dict_->words_[word_pos].back() == letter && b >= size)
	    {
	      age = size;
	      for (i = 0; i < size; ++i)
		age += age_hand_[dict_->words_[word_pos][i]] * 2;
	      matches->push_back(Action(&dict_->strings_[word_pos], &dict_->words_[word_pos], 
					&dict_->freqs_[word_pos], 'G', 
					(x - size * (1 - down) + grid_x_) % grid_x_, 
					(y - size * down + grid_y_) % grid_y_, down, age));
	    }
	}
    }

  if (grid_letters_[y][x] != -1)
    --freq_hand_[letter];
}

void Banjin::dumpable(std::vector<Action> *matches)
{
  if (pos_in_bunch_.size() < 2) 
    return;
  for (auto &pos : pos_in_hand_)
    matches->push_back(Action(&dict_->letter_strings_[tile_letter_[pos]], 
			      &dict_->letter_words_[tile_letter_[pos]], 
			      &dict_->letter_freqs_[tile_letter_[pos]], 
			      'D', tile_x_[pos], tile_y_[pos], false, 
			      age_hand_[tile_letter_[pos]]));
}


void Banjin::step()
{
  // Reset freqs and age up. We probably could automate freqs in grow/wilt instead
  for (uint32_t i = 0; i < dict_->alphabet_size_; ++i)
    {
      freq_growable_[i] = 0;
      freq_wiltable_[i] = 0;
      age_hand_[i] += freq_hand_[i];
      if (age_hand_[i] > max_age_)
	max_age_ = age_hand_[i];
    }

  // Peel if we don't have any tiles, as there's nothing else we'd do
  if (pos_in_hand_.size() == 0)
    peel();
  
#ifdef GRID_VERBOSE
  std::cout << *this;
#endif

  // Find growable words
  std::vector<Action> actions;
  Action *action = NULL;
  if (pos_in_grid_.size() > 0)
    {
      bool chance = dist(rng_) < 0.75;
      for (auto &pos : pos_in_grid_)
	if ((chance && is_end_tile(pos)) || (!chance && is_edge_tile(pos)))
	  growable(&actions, pos);
    }
  else
    growable(&actions, grid_x_ / 2, grid_y_ / 2, false, dict_->max_word_len_, dict_->max_word_len_);
  dumpable(&actions);      
  wiltable(&actions);

  // Select the action with the lowest value action
  std::sort(actions.begin(), actions.end(), [](const Action &l, const Action &r) -> bool { return l.get_value() > r.get_value(); });
  action = &actions[0];

  // Save the current state action pair
  events_.push_back(StateAction(pos_in_bunch_.size(), num_tiles_, dict_->alphabet_size_, freq_hand_, 
				freq_growable_, freq_wiltable_, age_hand_));
  events_.back().update_action(action->get_action(), &(*action->get_freq())[0]);
  
  switch (action->get_action())
    {
    case 'G': grow(action); break;
    case 'W': wilt(action); break;
    case 'D': dump(action); break;
    default: std::cerr << "WTF ACTION IS THIS\n";
    }
}

std::ostream &operator<<(std::ostream &output, const Banjin &b)
{
  output << "Seed: " << std::setw(20) << b.seed_ 
	 << " Cost: " << std::setw(4) << b.cost_ 
	 << " Dumps: " << std::setw(2) << b.num_dump_ 
	 << " Grows: " << std::setw(4) << b.num_grow_ 
	 << " Wilts: " << std::setw(4) << b.num_wilt_ << '\n';

#ifdef GRID_VERBOSE
  uint32_t i;
  int32_t x, y;

  // Print letter labels
  for (i = 0 ; i < b.dict_->alphabet_size_; ++i)
    output << std::setw(3) << b.dict_->alphabet_[i];
  output << '\n';

  // Bunch, Hand, Grid counts
  for (i = 0 ; i < b.dict_->alphabet_size_; ++i)
    output << std::setw(3) << b.freq_bunch_[i];
  output << std::setw(4) << b.pos_in_bunch_.size() << " Bunch\n";

  for (i = 0 ; i < b.dict_->alphabet_size_; ++i)
    output << std::setw(3) << b.freq_hand_[i];
  output << std::setw(4) << b.pos_in_hand_.size() << " Hand\n";

  for (i = 0 ; i < b.dict_->alphabet_size_; ++i)
    output << std::setw(3) << b.freq_grid_[i];
  output << std::setw(4) << b.pos_in_grid_.size() << " Grid\n";

  for (i = 0 ; i < b.dict_->alphabet_size_; ++i)
    output << std::setw(3) << b.freq_growable_[i];
  output << " Growable\n";

  for (i = 0 ; i < b.dict_->alphabet_size_; ++i)
    output << std::setw(3) << b.freq_wiltable_[i];
  output << " Wiltable\n";

  // Print out column digits
  output << "    ";
  for (x = 0 ; x < b.grid_x_; ++x)
    output << (x % 10);
  output << ' ';
  for (x = 0 ; x < b.grid_x_; ++x)
    output << (x % 10);
  output << "\n";

  // Print out tiles and neighbors
  for (y = 0 ; y < b.grid_y_; ++y)
    {
      output << std::setw(3) << y << ' ';
      for (x = 0; x < b.grid_x_; ++x)
        output << (char) (b.grid_letters_[y][x] == -1 ? ' ' : b.dict_->alphabet_[b.grid_letters_[y][x]]);
      output << ' ';
      for (x = 0; x < b.grid_x_; ++x)
	output << (int) b.grid_neighbors_[y][x];
      output << '\n';
    }
#endif
  return output;
}

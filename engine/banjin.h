#ifndef BANJIN_H
#define BANJIN_H

#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include "dictionary.h"
#include "action.h"
#include "state.h"

#define VERBOSE
#define GRID_VERBOSE

class Banjin
{
 protected:
  const int split_count_;
  const Dictionary *dict_;
  const uint64_t seed_;
  const int grid_x_;
  const int grid_y_;
  
  // Game state
  std::mt19937 rng_;
  std::uniform_real_distribution<double> dist;
  uint32_t cost_ = 0.0;

  int num_tiles_ = 0, num_dump_ = 0, num_wilt_ = 0, num_grow_ = 0, max_age_ = 0;
  int *tile_letter_;
  int *tile_x_;
  int *tile_y_;
  int *tile_in_bunch_;
  int *tile_in_hand_;
  int *tile_in_grid_;
  std::vector<int> pos_in_bunch_;
  std::vector<int> pos_in_hand_;
  std::vector<int> pos_in_grid_;
  std::vector<std::vector<int>> pos_in_hand_by_letter_;
  int **grid_letters_;
  int **grid_neighbors_;
  int **grid_positions_;
  int *freq_bunch_;
  int *freq_hand_;
  int *freq_grid_;
  int *freq_growable_;
  int *freq_wiltable_;
  int *age_hand_;
  std::vector<Action> words_;

  std::vector<StateAction> events_;

 public:
  Banjin(int split_count, const std::vector<int> *bunch_init, const Dictionary *dict, uint64_t seed=0, int grid_x=32, int grid_y=32);
  ~Banjin();

  void split();
  void peel();

  void give(int count);
  void take(int letter);

  void grow(Action *match);
  void wilt(Action *match);
  void dump(Action *match);


  void step();

  // Getters
  uint32_t get_cost() const { return cost_; }
  uint64_t get_seed() const { return seed_; }
  bool is_done() const;
  bool is_end_tile(int pos);
  bool is_edge_tile(int pos);
  
  int32_t y_b(int y) { return (y - 1 + grid_y_) % grid_y_; }
  int32_t y_a(int y) { return (y + 1) % grid_y_; }
  int32_t x_b(int x) { return (x - 1 + grid_x_) % grid_x_; }
  int32_t x_a(int x) { return (x + 1) % grid_x_; }

  void room(int x, int y, bool down, int *b, int *a);
  void wiltable(std::vector<Action> *matches);
  void growable(std::vector<Action> *matches, int x, int y, bool down, int b, int a);
  void growable(std::vector<Action> *matches, int pos);
  void dumpable(std::vector<Action> *matches);
  void peelable(std::vector<Action> *matches);

  // Operator overloading
  friend std::ostream &operator<<(std::ostream &output, const Banjin &b);
};

#endif

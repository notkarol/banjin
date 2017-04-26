#include "banjin.h" 
#include <limits>
#include <sqlite3.h>
#include <string.h>

typedef struct result
{
  uint64_t seed;
  uint32_t split;
  const char *dictionary;
  const char *bunch;
  float cost;
} result;

int save_to_db(std::vector<result> *results)
{
  // Database
  sqlite3 *db;
  char *zErrMsg = 0;
  char *sql = new char[512];
  sqlite3_open("results.db", &db);
  sqlite3_exec(db, "BEGIN TRANSACTION;", NULL, NULL, NULL);
  for (auto &result : *results)
    {
      sprintf(sql, "INSERT INTO results VALUES (%lu,%u,'%s','%s',%f);", 
	      result.seed, result.split, result.dictionary, result.bunch,  result.cost);
      if (sqlite3_exec(db, sql, NULL, 0, &zErrMsg) != SQLITE_OK) 
	{
	  std::cerr << "Insert got the error message: " << zErrMsg << "\n";
	  break;
	}
    }
  sqlite3_exec(db, "END TRANSACTION;", NULL, NULL, NULL);
  sqlite3_close(db);
  return 0;
}

void load_bunch(const char *input, std::vector<char> *alphabet, std::vector<int32_t> *bunch_freq)
{
  int32_t pos = 0;
  char c = '0';
  while ((c = input[pos]) > 0)
    {
      if (c >= '0' and c <= '9')
	{
	  bunch_freq->push_back(atoi(&input[pos]));
	  while (c >= '0' and c <= '9')
	    c = input[++pos];
	}
      else if (c > 0)
	{
	  alphabet->push_back(c);
	  ++pos;
	}
    }
}

int main(int argc, char *argv[])
{
  if (argc != 5)
    {
      std::cerr << "Need dictionary path, bunch, split size, number of games\n";
      exit(-1);
    }

  // Arguments
  const char* path = argv[1];
  const char* bunch = argv[2];
  uint32_t split = atoi(argv[3]);
  uint32_t loops = atoi(argv[4]);

  // Prepare random number genration
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());

  // Loop through banjins and save results
  Banjin *b;
  std::vector<result> results;
  std::vector<char> alphabet;
  std::vector<int32_t> bunch_freq;
  load_bunch(bunch, &alphabet, &bunch_freq);
  Dictionary *dictionary = new Dictionary(path, alphabet);
  float total_cost = 0;
  
  // Calculate the maximum limit. Assume we have 4 wilts per tile. 
  float max_cost = 0;
  for (uint32_t i = 0; i < bunch_freq.size(); ++i)
    max_cost += bunch_freq[i] * 5;

  for (uint32_t i = 0; i < loops; ++i)
    {
      b = new Banjin(split, &bunch_freq, dictionary, dist(rng));
      while (!b->is_done() && b->get_cost() < max_cost) {
	b->step();
      }
      results.push_back({b->get_seed(), split, path, bunch, b->get_cost() / max_cost});
      std::cout << *b;
      total_cost += results.back().cost;
      delete b;
    }
  std::cout << total_cost / loops << "\n"; 
  delete dictionary;
  return save_to_db(&results);
}

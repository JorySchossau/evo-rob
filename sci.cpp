#include "pprint.hpp" // python-like print()
#include "CLI11.hpp" // cli parameters
#include "archive.h" // serialization
#include "profiling.h" // for code profiling / timing
#include <armadillo> // matlab-like fast math

#include <vector>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <ctime> // time() / srand
#include <csignal> // for ctrl-c signal
#include <unordered_set>
#include <queue>
#include <fstream> // ifstream, ofstream
#include <string>
#include <iterator>
#include <random>

#ifdef _WIN32
  #include <process.h> // process ID
#else /* linux */
  #include <unistd.h> // process ID
#endif

// deterministic but fast generator
// should probably replace the randf()
// with this one because it's faster
// than the underyling C rand() by a lot
// and C rand() is not x-platform deterministic.
std::mt19937_64 generator(34);
std::uniform_real_distribution<float> real_dist(0.0f,1.0f);
std::uniform_int_distribution<int> int_dist(0,INT_MAX);
static inline float drandf() {
  return real_dist(generator);
}
static inline int drand() {
  return int_dist(generator);
}
static inline void dsrand(const unsigned int& newseed) {
  generator.seed(newseed);
}

/* return a seed and also use it as the new seed */
static inline unsigned int drandGetSeed() {
  unsigned int seed = generator();
  generator.seed(seed);
  return seed;
}

#ifdef NO_BOUNDS_CHECKING
// safety first :)
//#define at operator[] // stl-specific version (use .at everywhere in stl and use this to get performance when ready)
#define ARMA_NO_DEBUG 1 // armadillo-specific version (use () indexing in armadillo and then use this to get performance when ready)
#endif

// set up better cout as a python-like print()
// now we can use `print(anything,...);`
using namespace pprint;
pprint::PrettyPrinter printer;
#define print printer.print

// convenience for begin, end
using std::begin, std::end;

// set up armadillo lib math experience
// now we can do linear algebra
using namespace arma;

// set up faster rand fn
// much faster random num from 0 to 1 than using
// "high quality" mersenne twister or armadillo lib.
#define randf() float(rand())/float(RAND_MAX)

namespace arma::fill {
  // the arma version of filling with random numbers is very slow compared to C's rand fn.
  // here, we make a templated fn similar to arma's that does things much faster.
  // we aren't doing statistical physics or cryptography, so we don't care about
  // the quality of the random distribution, so we can just use C's rand fn.
  // example: fill::random(my_arma_container);
  template <typename C, typename T = typename C::value_type, typename = T>
  void random(C& vec) {
    //std::generate(begin(vec), end(vec), [] { return (T)std::rand() / RAND_MAX; });
    std::generate(begin(vec), end(vec), [] { return (T)drandf(); });
  }
}

// temp pasted

/** globals **/
namespace GLB {
  /* EVOLVE command parameters */
  // evolution
  unsigned int screen_update_interval = 500;
  unsigned int generations_limit = 1'000;
  unsigned int lod_save_interval = 1'000;
  unsigned int current_generation; // used in main() loop
  int seed {-1};
  int pop_size = 100;
  int G = 40;
  float point_mutation_rate = 0.005f;
  float vector_mutation_rate = 0.0005f;
  bool use_point_mu = true;
  bool use_col_mu = true;
  bool use_row_mu = true;
  bool local_locality_mu = false;
  int developmental_updates = 200;
  std::vector<int> hist_gens;
  // fitness landscape
  int N=20;
  int K=1;
  float speed_change = 0.001;
  /* overlap */
  // saving, loading
  std::string savefilename{""}, loadfilename{""};
  /* ANALYZE command parameters */
  std::vector<int> gen_pick {0,1,-1};
  unsigned int ntrials {500};
  unsigned int maxmutations {5};
  bool show_fitness {false};
  bool show_robustness {false};
  bool show_phenotype {false};
  /* INFO command parameters */
}

/** GRN **/
class GRN {
  public:
    //this is a dampening constant, leave it where it is
    const static inline float GRN_DAMP {0.1f};
  public:
    int num_genes;
    fmat network;
    frowvec state, projected_state;
    GRN() = delete;
    GRN(const fmat& /*genome*/);
    void updateNetwork(const int& iterations=1);
};

GRN::GRN(const fmat& matrix) : num_genes(size(matrix)[0]) {
  state.resize(num_genes);
  projected_state.resize(num_genes);
  network = matrix;
  // fill with values of 1/SIZE
  state = ones<frowvec>(num_genes) / num_genes;
  // constrain values and make into unit-vector
  state = clamp(state, 0.0001, 1.0);
  state /= accu(state);
}

void GRN::updateNetwork(const int& iterations) {
  for (int iteration=iterations-1; iteration>=0; --iteration) {
    projected_state = state * network;
    projected_state = (2.0f/(exp(-projected_state)+1.0f)) - 1.0f;
    state -= (state-projected_state) * GRN_DAMP;
  }
}

/** Agent Code **/

// recording genomic diffs
struct GenomeChange {
  public:
    int location;
    float newvalue;
};

/* We make mu and num_genes not constants
 * but runtime definable variables
 * to support loading data and rerunning.
 */
struct AgentClassConfiguration {
  public:
    int num_genes;
    float mu_point;
    float mu_vector;
};

// used for specifying in the Agent ctor
// if it should be randomly initialized or not.
enum INIT {NONE=0, RANDOM=1};

/** Agent **/
class Agent {
  public:
    static inline int num_genes {-1}; // default to invalid value
    static inline float mu {-0.0001f}; // default to invalid value
    // must be called before using the class.
    static void configure(const AgentClassConfiguration& config);
    static inline std::default_random_engine mu_generator{};
    static inline std::binomial_distribution<int> mu_point_distribution{}, mu_vector_distribution{};

  public:
    float fitness{-1};
    fmat genome; // (serialized)
    bool* phenotype {nullptr}; // continuous genome -> continuous grn -> boolean phenotype
    bool developed {false};
    unsigned int gen_of_evaluation; // (serialized) (needed for reconstructing env::treadmill)
    std::unique_ptr<GRN> grn {nullptr};
    std::shared_ptr<Agent> ancestor {nullptr};
    std::shared_ptr<Agent> progenitor {nullptr}; // only used during LOD serialization
    std::vector<GenomeChange> changeset; // (serialized) tracks mutations
    unsigned int new_mutations {0}; // usually same as changeset.size(), unless picking sparsely from LOD, then will be larger

    Agent(); // alias for Agent(INIT::NONE);
    ~Agent(); // alias for Agent(INIT::NONE);
    Agent(const INIT& /*init*/); // determines how the agent is initialized
    void inheritFrom(const std::shared_ptr<Agent> /*parent*/); // inherit from another agent asexually
    void initGRN();
    void runGRN(const int& num_updates);
    void developPhenotypeFromGRN();
    void mutatePointWise();
    void mutateColWise();
    void mutateRowWise();
};

/*
 * Parameters used by all agents.
 * gene number,
 * mutation rate,
 * and mutation distribution generator.
 */
void Agent::configure(const AgentClassConfiguration& config) {
  Agent::num_genes = config.num_genes;
  mu_point_distribution.param(std::binomial_distribution<int>::param_type(Agent::num_genes*Agent::num_genes, config.mu_point));
  mu_vector_distribution.param(std::binomial_distribution<int>::param_type(Agent::num_genes, config.mu_vector));
}

Agent::Agent() : Agent(INIT::NONE) { }

Agent::Agent(const INIT& init) {
  genome.resize(num_genes,num_genes);
  switch(init) {
    case INIT::NONE:
      break;
    case INIT::RANDOM:
      // fill with random values [0,1] then stretch to [-1,1]
      fill::random(genome);
      genome = genome*2 - 1;
      break;
  }
}

Agent::~Agent() {
  if (phenotype) delete [] phenotype;
}

void Agent::mutatePointWise() {
  // determine number of mutations
  // that should happen,
  // and return early if 0.
  // if no mutations, then
  // return early.
  int num_mutations = Agent::mu_point_distribution(Agent::mu_generator);
  if (num_mutations == 0) { return; }
  developed = false;
  int genome_size = num_genes * num_genes;
  int pos; // flattened matrix index position
  float mutation_value;
  while (num_mutations > 0) {
    pos = drand() % genome_size;
    if (GLB::local_locality_mu) {
      mutation_value = randf()*0.1f - 0.05f;
      genome(pos) += mutation_value; // local-locality mutation
      changeset.emplace_back(GenomeChange{
                             .location = pos,
                             .newvalue = genome(pos)
                             });
    } else {
      mutation_value = randf()*2-1;
      mutation_value = std::clamp(mutation_value, 0.0f, 1.0f);
      genome(pos) = mutation_value; // global-locality mutation
      changeset.emplace_back(GenomeChange{
                             .location = pos,
                             .newvalue = mutation_value
                             });
    }
    --num_mutations;
  }
}

void Agent::mutateColWise() {
  int num_mutations = Agent::mu_vector_distribution(Agent::mu_generator);
  if (num_mutations == 0) { return; }
  developed = false;
  int col_size = num_genes;
  int pos; // flattened matrix index position
  fcolvec col_offset_values(col_size);
  while (num_mutations > 0) {
    pos = drand() % col_size;
    // make random offsets in [-0.1,0.1]
    fill::random(col_offset_values);
    col_offset_values -= 0.5;
    col_offset_values *= 0.2;
    genome.col(pos) += col_offset_values;
    // record each change into changeset
    for (int i=col_size-1; i>=0; --i) {
      changeset.emplace_back(GenomeChange{
                             .location = (pos*col_size)+i,
                             .newvalue = genome(pos*col_size+i)
                             });
    }
    --num_mutations;
  }
}

void Agent::mutateRowWise() {
  int num_mutations = Agent::mu_vector_distribution(Agent::mu_generator);
  if (num_mutations == 0) { return; }
  developed = false;
  int row_size = num_genes;
  int pos; // flattened matrix index position
  frowvec row_offset_values(row_size);
  while (num_mutations > 0) {
    pos = drand() % row_size;
    // make random offsets in [-0.1,0.1]
    fill::random(row_offset_values);
    row_offset_values -= 0.5;
    row_offset_values *= 0.2;
    genome.row(pos) += row_offset_values;
    // record each change into changeset
    for (int i=row_size-1; i>=0; --i) {
      changeset.emplace_back(GenomeChange{
                             .location = (pos*row_size)+i,
                             .newvalue = genome(pos*row_size+i)
                             });
    }
    --num_mutations;
  }
}

void Agent::inheritFrom(const std::shared_ptr<Agent> parent) {
  genome = parent->genome;
  ancestor = parent;
  developed = true;
  if (GLB::use_point_mu) mutatePointWise();
  if (GLB::use_col_mu) mutateColWise();
  if (GLB::use_row_mu) mutateRowWise();
  if (developed) {
    if (not phenotype) { phenotype = new bool[GLB::N]; }
    for (int i=0; i<GLB::N; i++) phenotype[i] = parent->phenotype[i];
  }
}

auto Agent::initGRN() -> void {
  grn = std::make_unique<GRN>(genome);
}

auto Agent::runGRN(const int& num_updates) -> void {
  grn->updateNetwork(num_updates);
}

/** selection methods **/
namespace SELECTION {
  /* complete_roulette performs fitness-proportional
   * selection, replacing the entire population
   * at each generation, inheriting from the previous
   * generation.
   */
  void complete_roulette(std::vector<std::shared_ptr<Agent>>& oldpop, std::vector<std::shared_ptr<Agent>>& newpop, const fcolvec& fitnesses, const float& minW, const float& maxW) {
    int popsize = oldpop.size();
    int parent;
    // if no selection gradient, then pick randomly
    /* [[ unlikely ]] */
    if (minW == maxW) {
      for (int i=0; i<popsize; i++) {
        parent = drand() % popsize;
        newpop.emplace_back( std::make_shared<Agent>(INIT::NONE) );
        newpop.back()->inheritFrom( oldpop[parent] );
      }
    // if selection gradient, then pick fitness proportional (stochastic)
    } else /*(minW != maxW)*/ {
      fcolvec W = (fitnesses - minW) / (maxW - minW);
      for (int i=0; i<popsize; i++) {
        do {
          parent = rand() % popsize;
        } while (randf()>W(parent));
        newpop.emplace_back( std::make_shared<Agent>(INIT::NONE) );
        newpop.back()->inheritFrom( oldpop[parent] );
      }
    }
  }
}

auto Agent::developPhenotypeFromGRN() -> void {
  if (not phenotype) { phenotype = new bool[GLB::N]; }
  for (int i=0; i<GLB::N; ++i) { phenotype[i] = (grn->state[i]>0.0f); }
}

/** fitness functions **/
namespace ENV {
  namespace MAXONES {
    auto evaluate(std::shared_ptr<Agent> agent) -> float {
      return accu(clamp(agent->genome,0,1));
    }
  }
  namespace NKTREADMILL {
    /* helper function for fitness
     * evaluation.
     */
    float triangleSin(float x, int resolution){
      float Y=0.0;
      for(int i=0;i<GLB::N;i++){
        float n=(2*i)+1;
        Y+=std::pow(-1,float(i))*std::pow(n,-2.0f)*std::sin(n*x);
      }
      return (0.25f*float{M_PI})*Y;
    }

    /* initialize alphas and betas
     * tables. Set and store random
     * number seed before this.
     */
    fmat Alphas,Betas;
    unsigned int alpha_beta_seed {0};
    auto init(const unsigned int& new_seed) -> void {
      Alphas.set_size(GLB::N,1<<GLB::K);
      Betas.set_size(GLB::N,1<<GLB::K);
      dsrand(new_seed);
      alpha_beta_seed = new_seed;
      fill::random(Alphas);
      fill::random(Betas);
    }

    /* call init() before this,
     * call agent's developPhenotypeFromGRN() before this.
     */
    auto evaluate(std::shared_ptr<Agent> agent) -> float {
      float W=0.0;
      for(int n=0;n<GLB::N;n++){ // ex: 0-20
        int val=0;
        for(int k=0;k<GLB::K;k++){ // ex: 0-3
          val=(val<<1)+agent->phenotype[(n+k)%GLB::N]; // phenotype is bool data
        }
        W+=std::log((1.0f+ENV::NKTREADMILL::triangleSin(((GLB::speed_change*agent->gen_of_evaluation)*(Betas(n,val)+0.5f))+(Alphas(n,val)*float{M_PI}*2.0f),5))/2.0f);
      }
      return std::exp(W/float(GLB::N));
    }

    /* getMaxFitAgent
     * Returns max fit ideal organism.
     * Enumerates genomes to find.
     * fitness peak (for generation 0)
     */
    auto getMaxFitAgent() -> std::shared_ptr<Agent> {
      std::shared_ptr<Agent> A = std::make_shared<Agent>(INIT::NONE);
      int bestg;
      A->phenotype = new bool[GLB::N];
      A->gen_of_evaluation = 0;
      float globalPeak=0.0f, fitness=0.0f;
      for(int g=0;g<(1<<GLB::N);g++){
        for(int j=0;j<GLB::N;j++) { A->phenotype[j] = (g>>j)&1; }
        fitness = ENV::NKTREADMILL::evaluate(A);
        if(fitness>globalPeak) {
          globalPeak=fitness;
          bestg = g;
        }
      }
      // fill in fitness and phenotyp only
      std::shared_ptr<Agent> bestAgent = std::make_shared<Agent>(INIT::NONE);
      bestAgent->fitness = globalPeak;
      bestAgent->phenotype = new bool[GLB::N];
      bestAgent->gen_of_evaluation = 0;
      for(int j=0;j<GLB::N;j++) { bestAgent->phenotype[j] = (bestg>>j)&1; }
      return bestAgent;
    }
  }
}

/** LOD functions **/
namespace LOD {
  /* Finds the Most Recent Common Ancestor,
   * returns a tuple (pair) with a pointer to
   * the ancestor, and how many generations
   * back we had to crawl to find it. [0-N)
   */
  auto getMRCA(std::vector<std::shared_ptr<Agent>> population) -> std::pair< std::shared_ptr<Agent>, unsigned int> {
      std::queue<std::shared_ptr<Agent>> mrca_queue;
      std::queue<int> mrca_queue_gen_sizes;
      std::unordered_set<std::shared_ptr<Agent>> mrca_set_prev_gen;
      std::shared_ptr<Agent> mrca;
      std::shared_ptr<Agent> ancestor;
      // fill queue with initial population
      mrca_queue_gen_sizes.push(population.size());
      for (auto& member : population) {
        mrca_queue.push(member);
      }
      // begin traversing queue
      bool stop_search {false};
      int generations_back {0};
      while (true) {
        // go through n members of the queue 
        // for n members who were together 
        // in the same generation, because
        // the queue can hold more members
        // from the next generation, too.
        mrca_set_prev_gen.clear();
        int generation_memberi = mrca_queue_gen_sizes.front()-1; mrca_queue_gen_sizes.pop();
        for ( ; generation_memberi>=0; --generation_memberi) {
          ancestor = mrca_queue.front(); mrca_queue.pop();
          //look at grandparent, then add to set if not null
          // if found null then there is no mrca
          if (ancestor->ancestor == nullptr) { return std::make_pair(nullptr,generations_back); }
          mrca_set_prev_gen.insert(ancestor->ancestor);
        }
        // stop and return if we're at a common ancestor
        if (mrca_set_prev_gen.size() == 1) {
          mrca = *begin(mrca_set_prev_gen);
          return std::make_pair(mrca,generations_back);
        }
        // after adding all grandparents to the set,
        // go through the unique set and add them to queue
        for (auto& ancestor : mrca_set_prev_gen) {
          mrca_queue.push(ancestor);
        }
        mrca_queue_gen_sizes.push(mrca_set_prev_gen.size());
        ++generations_back;
      }
      // should never hit
      return std::make_pair(nullptr,generations_back);
    }

  /* serialize the LOD below the mrca
   * to a file (not mrca itself).
   */
  auto save(const std::string& filename, std::shared_ptr<Agent> mrca) -> void {
    std::ofstream touchFS(filename,ios::out|ios::app); touchFS.close(); // necessary
    std::fstream outFS(filename,ios::binary|ios::in|ios::out);
    outFS.seekp(0,ios::end);
    if (not outFS.is_open()) {
      std::cerr << "Error: '" << filename << "' can't be opened." << std::endl;
      exit(1);
    }
    std::shared_ptr<Agent> current, mdca; // most distant common ancestor
    // crawl LOD to find how many there will be.
    // nothing ot be done if we're already at the end
    if (mrca->ancestor == nullptr) return;
    // format:
    // header: (only if stream position is 0 - start of file)
    // generations (int) // num genes (int)
    // first genome in LOD (floats)
    // body:
    // num changes in this generation (int)
    // [for each change] // location (int) // new value (float)
    current = mrca;
    int generations_count{0};
    while (current->ancestor not_eq nullptr) {
      // make forward link
      current->ancestor->progenitor = current;
      // crawl to next one and count
      current = current->ancestor;
      ++generations_count;
    }
    mdca = current;
    Archive<std::fstream> serialize(outFS);

    // HEADER only if at pos 0 in file
    // serialize the initial genome of
    // most distant common ancestor on
    // remaining LOD.
    if (outFS.tellp() == 0) {
      serialize << generations_count;
      serialize << ENV::NKTREADMILL::alpha_beta_seed;
      serialize << GLB::N << GLB::K << GLB::G;
      serialize << GLB::speed_change; // rate of change
      serialize << Agent::num_genes;
      for (auto& e : mdca->genome) { serialize << e; }
      current = current->progenitor;
    } else {
      // update the generations count at front of file
      // (newvalue = oldvalue + morevalue)
      int previous_count;
      outFS.seekp(0);
      serialize >> previous_count; // read old
      generations_count += previous_count; // calculate new
      outFS.seekp(0);
      serialize << generations_count; // write new
      outFS.seekp(0,ios::end); // go back to appending
    }
    // proceed back up from mdca using forward links.
    // crawl the LOD and serialize each changeset
    int changeset_size;
    while (current->progenitor not_eq nullptr) {
      changeset_size = current->changeset.size();
      serialize << changeset_size;
      for (auto& change : current->changeset) {
        serialize << change.location << change.newvalue;
      }
      current = current->progenitor;
    }
    outFS << std::flush;
    outFS.close();
  }

  /* clean up (delete) the LOD after
   * a common ancestor. Typically, this
   * is the MRCA.
   */
  auto pruneAt(std::shared_ptr<Agent> mrca) -> void {
    std::shared_ptr<Agent> current = mrca;
    // get to mdca
    while (current->ancestor not_eq nullptr) {
      current->ancestor->progenitor = current;
      current = current->ancestor;
    }
    // now walk back up, clearing pointers
    while ( (current->progenitor not_eq nullptr) and (current->progenitor not_eq mrca) ) {
      current = current->progenitor;
      current->ancestor->progenitor = nullptr;
      current->ancestor = nullptr;
    }
    current->progenitor = nullptr;
    mrca->ancestor = nullptr;
  }

  /* Utility fn to find the depth of a unidirectional
   * linked list (lod). I don't think I use this anymore.
   */
  auto getDepthBelow(std::shared_ptr<Agent> mrca) -> int {
    std::shared_ptr<Agent> current = mrca;
    int depth {0};
    while (current->ancestor not_eq nullptr) {
      current = current->ancestor;
      ++depth;
    }
    return depth;
  }

  // specifier when loading lod file
  enum RECONSTRUCTION {SPARSE=0, FULL};

  /* Load lod data from disk.
   * Takes filename.
   * only loads generations specified by --pick [start,skip,end]
   * returns vector of only those lod generations.
   */
  auto load(const std::string& filename) -> std::vector<std::shared_ptr<Agent>> {
    std::ifstream inFS(filename,ios::binary|ios::in);
    if (not inFS.is_open()) {
      std::cerr << "Error: '" << filename << "' can't be opened." << std::endl;
      exit(1);
    }
    std::vector<std::shared_ptr<Agent>> lod;
    Archive<std::ifstream> deserialize(inFS);
    // load generations, gene count, first genome
    int generations, gene_count;
    deserialize >> generations;
    // if no one supplied end, then pick very end
    if (GLB::gen_pick.back() == -1) GLB::gen_pick.back() = generations;
    if (GLB::gen_pick.front() == -1) GLB::gen_pick.front() = generations;
    unsigned int space_to_reserve = (GLB::gen_pick.back()-GLB::gen_pick.front())/GLB::gen_pick[1]+1;
    lod.reserve(space_to_reserve);
    deserialize >> ENV::NKTREADMILL::alpha_beta_seed;
    deserialize >> GLB::N >> GLB::K >> GLB::G;
    deserialize >> GLB::speed_change; // rate of change
    deserialize >> gene_count;
    GLB::G = gene_count;
    Agent::configure({ .num_genes=GLB::G, .mu_point=GLB::point_mutation_rate/(GLB::G*GLB::G), .mu_vector=GLB::vector_mutation_rate/GLB::G });
    int network_size = gene_count*gene_count;
    lod.emplace_back(std::make_shared<Agent>(INIT::NONE));
    lod.back()->genome.resize(gene_count,gene_count);
    float genome_value;
    // read first genome
    for (int i=0; i<network_size; i++) {
      deserialize >> genome_value;
      lod.back()->genome(i) = genome_value;
    }
    int numchanges, location;
    float newvalue;
    // update first genome to start of picked generations, if they are different
    for (int i=1; i<GLB::gen_pick.front(); i++) {
      deserialize >> numchanges;
      lod[0]->new_mutations += numchanges;
      lod[0]->gen_of_evaluation = i;
      for (int changei=0; changei<numchanges; ++changei) {
        deserialize >> location >> newvalue;
        lod[0]->genome(location) = newvalue;
      }
    }
    // gen_pick [start,stride,end]
    for (int i=GLB::gen_pick.front()+GLB::gen_pick[1]; i<=GLB::gen_pick.back(); i+=GLB::gen_pick[1]) {
      lod.emplace_back(std::make_shared<Agent>(INIT::NONE));
      lod.back()->genome = lod[lod.size()-2]->genome;
      lod.back()->gen_of_evaluation = i;
      for (int ii=i-GLB::gen_pick[1]; ii<i; ++ii) {
        deserialize >> numchanges;
        lod.back()->new_mutations += numchanges;
        for (int changei=0; changei<numchanges; ++changei) {
          deserialize >> location >> newvalue;
          lod.back()->genome(location) = newvalue;
        }
      }
    }
    inFS.close();
    return lod;
  }
}

namespace ANALYSIS {

  auto inline showWDist(std::vector<std::shared_ptr<Agent>>& pop, const float& minval, const float& maxval, const int& bins) -> void {
    std::vector<int> dist(bins, 0);
    for (auto& agent : pop) {
      ++dist[int(floor(agent->fitness/maxval*(bins-1)))];
    }
    std::cout << std::string(bins,'=') << std::endl;
    int height = int(floor((*std::max_element(begin(pop), end(pop), [&](std::shared_ptr<Agent>& a, std::shared_ptr<Agent>& b){return a->fitness < b->fitness;}))->fitness/maxval*(bins-1)));
    while (height > 0) {
      for (int i=0; i<bins; i++) {
        if (dist[i] >= height) std::cout << '*';
        else                   std::cout << ' ';
      }
      std::cout << std::endl;
      --height;
    }
    std::cout << std::string(bins,'=') << std::endl;
  }

  struct RobustnessConfiguration {
    public:
    const std::shared_ptr<Agent>& agent;
    const unsigned int& maxmutations;
    const unsigned int& ntrials;
    const unsigned int& generation;
    const std::function<float(std::shared_ptr<Agent>)>& fitness_fn;
  };

  auto inline getRobustness(const RobustnessConfiguration& cfg) -> float {
    uvec mu_locs;
    fcolvec mu_newvals;
    fcolvec mu_oldvals;
    fmat& matrix = cfg.agent->genome;
    int genome_size = matrix.n_elem;
    std::shared_ptr<Agent> agent = cfg.agent;
    agent->gen_of_evaluation = cfg.generation;

    fcolvec avg_robustness(cfg.maxmutations); // the 'curve' of robustness
    fcolvec w(cfg.ntrials); // filled each detail level
    for (int num_mutations=1; num_mutations<=cfg.maxmutations; num_mutations++) {
      mu_locs.set_size(num_mutations);
      mu_newvals.set_size(num_mutations);
      mu_oldvals.set_size(num_mutations);
      for (int i=0; i<cfg.ntrials; i++) {
        // make mutations of matrix
        std::generate_n(begin(mu_locs), num_mutations, [&](){return drand()%genome_size;});
        fill::random(mu_newvals);
        mu_oldvals = matrix(mu_locs);
        matrix(mu_locs) = mu_newvals;
        // evaluate agent
        agent->initGRN();
        agent->runGRN(GLB::developmental_updates);
        agent->developPhenotypeFromGRN();
        w(i) = cfg.fitness_fn(agent);
        // reset matrix values
        matrix(mu_locs) = mu_oldvals;
      }
      // record avg robustness for particular mutation sample size (num_mutations-1)
      avg_robustness(num_mutations-1) = mean(w);
    }
    // return accumulation under the "robustness curve" for this individual
    return accu(avg_robustness);
  }
}

namespace RUN {
  /* evolution loop for running nk treadmill
   * with specified fitness function.
   */
  auto grn_nktreadmill_haploid_evolution(const std::function<float(std::shared_ptr<Agent>)> fitness_fn) {
    // init code
    auto seed = (GLB::seed == -1) ? getpid() : GLB::seed;
    print("rand_seed:",seed);
    srand(seed); dsrand(seed);
    Agent::configure({ .num_genes=GLB::G, .mu_point=GLB::point_mutation_rate/(GLB::G*GLB::G), .mu_vector=GLB::vector_mutation_rate/GLB::G });
    ENV::NKTREADMILL::init(seed);

    // zero out save file
    if (GLB::savefilename.size() > 0) {
      std::ofstream outFS(GLB::savefilename,ios::binary|ios::trunc|ios::out);
      if (not outFS.is_open()) {
        std::cerr << "Error: couldn't open '"+GLB::savefilename+"'" << std::endl;
        exit(1);
      }
      outFS.close();
    }

    std::vector<std::shared_ptr<Agent>> population, newpopulation;
    fcolvec fitnesses(GLB::pop_size);

    // create population
    for (int i=0; i<GLB::pop_size; i++) population.emplace_back(std::make_shared<Agent>(INIT::RANDOM));

    // loop generations
    float minW(FLT_MAX),maxW(FLT_MIN),meanW(0);
    for (GLB::current_generation=0; GLB::current_generation<GLB::generations_limit+1; GLB::current_generation++) {
      // evaluate population
      minW = FLT_MAX; maxW = FLT_MIN; meanW = 0.0f;
      for (int i=0; i<GLB::pop_size; i++) {
        if (not population[i]->developed) {
          population[i]->initGRN();
          population[i]->runGRN(GLB::developmental_updates);
          population[i]->developPhenotypeFromGRN();
        }
        population[i]->gen_of_evaluation = GLB::current_generation;

        // fitness functions
        //fitnesses(i) = ENV::MAXONES::evaluate(population[i]);
        //fitnesses(i) = ENV::NKTREADMILL::evaluate(population[i]);
        fitnesses(i) = fitness_fn(population[i]);

        population[i]->fitness = fitnesses(i);
        meanW += fitnesses(i);
        if (minW > fitnesses(i)) minW = fitnesses(i);
        if (maxW < fitnesses(i)) maxW = fitnesses(i);
      }

      // display running progress in intervals
      if ((GLB::current_generation % GLB::screen_update_interval)==0) {meanW /= GLB::pop_size; print(GLB::current_generation,'\t',meanW,'\t',maxW);}

      // display histogram
      if ( (GLB::hist_gens.size()>0) && (GLB::current_generation == GLB::hist_gens[0]) ) {
        ANALYSIS::showWDist(population,0,1,60);
        GLB::hist_gens.erase(begin(GLB::hist_gens), begin(GLB::hist_gens)+1);
      }

      // save and prune LOD in intervals
      if (GLB::savefilename.size() > 0) {
        if ((GLB::current_generation % GLB::lod_save_interval)==0) {
          auto [mrca, gens_back] = LOD::getMRCA(population);
          if (mrca not_eq nullptr) {
            LOD::save(GLB::savefilename, mrca);
            LOD::pruneAt(mrca);
          }
        }
      }

      // select new population
      SELECTION::complete_roulette(population, newpopulation, fitnesses, minW, maxW);
      // generation progression
      population.clear();
      population.swap(newpopulation);
    }
  }

  /* Analyzes the robustness of a lod
   * Produces 2 columns: generation robustness
   */
  auto analyze(const std::function<float(std::shared_ptr<Agent>)>& fitness_fn) -> void {
    // load() populates ENV::NKTREADMILL::alpha_beta_seed
    // load() only loads generations specified by --pick command line option [start,skip,end]
    // with default [0,1,-1] (where -1 indicates end)
    auto lod = LOD::load(GLB::loadfilename); // std::vector<std::shared_ptr<Agent>>
    ENV::NKTREADMILL::init(ENV::NKTREADMILL::alpha_beta_seed);
    std::ofstream outFS;
    if (GLB::savefilename.size() > 0) { outFS.open(GLB::savefilename,ios::out|ios::trunc); }
    
    // find best agent
    std::shared_ptr<Agent> best = ENV::NKTREADMILL::getMaxFitAgent();
    std::cout << "best phenotype: ";
    for (int i=0; i<GLB::N; i++) std::cout << best->phenotype[i];
    std::cout << endl;

    // print headers
    std::cout << "generation"; if (GLB::show_fitness) std::cout << ",fitness,norm_fitness"; if (GLB::show_robustness) std::cout << ",robustness"; if (GLB::show_phenotype) std::cout << ",phenotype";
    std::cout << std::endl;
    if (outFS.is_open()) {
      outFS << "generation"; if (GLB::show_fitness) outFS << ",fitness,norm_fitness"; if (GLB::show_robustness) outFS << ",robustness"; if (GLB::show_phenotype) outFS << ",phenotype";
      outFS << std::endl;
    }
    // loop through picked gens
    for (int i=0; i<lod.size(); ++i) {
      // reconstruct genome i
      lod[i]->initGRN();
      lod[i]->runGRN(GLB::developmental_updates);
      lod[i]->developPhenotypeFromGRN();
      std::cout << lod[i]->gen_of_evaluation;
      if (outFS.is_open()) outFS << lod[i]->gen_of_evaluation;
      // fitness inspection
      if (GLB::show_fitness) {
        // fitness functions
        //fitnesses(i) = ENV::MAXONES::evaluate(population[i]);
        //fitnesses(i) = ENV::NKTREADMILL::evaluate(population[i]);
        float w = fitness_fn(lod[i]);
        std::cout << "," << w << "," << w/best->fitness;
        if (outFS.is_open()) outFS << "," << w << "," << w/best->fitness;
      }
      // robustness inspection
      if (GLB::show_robustness) {
        float rmean = ANALYSIS::getRobustness({ .agent=lod[i], .maxmutations=GLB::maxmutations, .ntrials=GLB::ntrials, .generation=lod[i]->gen_of_evaluation, .fitness_fn=fitness_fn });
        std::cout << "," << rmean;
        if (outFS.is_open()) outFS << "," << rmean;
      }
      // phenotype inspection
      if (GLB::show_phenotype) {
        std::cout << ",";
        if (outFS.is_open()) outFS << ",";
        for (int n=0; n<GLB::N; n++) {
          std::cout << lod[i]->phenotype[n];
          if (outFS.is_open()) outFS << lod[i]->phenotype[n];
        }
      }
      // finish line
      std::cout << std::endl;
      if (outFS.is_open()) outFS << std::endl;
    }
    if (outFS.is_open()) outFS.close();
  }

  /* Reports information about a lod file */
  auto info() -> void {
    // set selection to start and finish at end of lod
    // thereby only recreating the genome and other
    // information for 1 agent (the final one)
    GLB::gen_pick.front() = -1;
    GLB::gen_pick.back() = -1;
    // load() automatically populates ENV::NKTREADMILL::alpha_beta_seed
    auto lod = LOD::load(GLB::loadfilename); // std::vector<std::shared_ptr<Agent>>
    print("lod size:",lod[0]->gen_of_evaluation+1);
    print("lod mutations:",lod[0]->new_mutations, "(average",ceil(float(lod[0]->new_mutations)/lod[0]->gen_of_evaluation*100)/100,"per generation)");
    print("GRN genes:",GLB::G);
  }
}

int main(int argc, char* argv[]) {
  CLI::App app{"Evolvability Science Tool"}; // shrug
  app.require_subcommand(1); // allow only 1 subcommand (evolve, analyze, info)
  CLI::App* sc_evolve = app.add_subcommand("evolve","evolve a population");
  CLI::App* sc_test = app.add_subcommand("test","evolve a population on Max Ones");
  CLI::App* sc_analyze = app.add_subcommand("analyze","analyze a line of descent");
  CLI::App* sc_info = app.add_subcommand("info","show information about a line of descent");

  /* evolve subcommand */
  for (auto& sc : {sc_evolve,sc_test}) {
    sc->add_option("-p,--popsize",GLB::pop_size,"population size ["+std::to_string(GLB::pop_size)+"]")->check(CLI::Range(0,INT_MAX));
    sc->add_option("-N",GLB::N,"NK (N) 'genome' length")->check(CLI::Range(0,INT_MAX))->required();
    sc->add_option("-K",GLB::K,"NK (K) 'gene' length")->check(CLI::Range(0,INT_MAX))->required();
    sc->add_option("-G",GLB::G,"GRN (G) number of 'genes' (mat GxG)")->check(CLI::Range(0,INT_MAX))->required();
    sc->add_option("-d,--dev-update",GLB::developmental_updates,"number of developmental updates ["+std::to_string(GLB::developmental_updates)+"]");
    sc->add_option("--rate",GLB::speed_change,"rate of environmental change ["+std::to_string(GLB::speed_change)+"]");
    sc->add_option("-g,--gen",GLB::generations_limit,"number of generations ["+std::to_string(GLB::generations_limit)+"]")->check(CLI::Range(0,INT_MAX));
    sc->add_option("-s,--screen-update",GLB::screen_update_interval,"stats screen update interval ["+std::to_string(GLB::screen_update_interval)+"]")->check(CLI::Range(0,INT_MAX));
    sc->add_option("-l,--lod-update",GLB::lod_save_interval,"lod saving/prune interval ["+std::to_string(GLB::lod_save_interval)+"]")->check(CLI::Range(0,INT_MAX));
    sc->add_option("--hist-gens",GLB::hist_gens,"csv which gens to show hist")->delimiter(',');
    sc->add_option("--mup",GLB::point_mutation_rate,"genomic point-wise mutation rate ["+std::to_string(GLB::point_mutation_rate)+"]")->check(CLI::Bound(0.0,1.0));
    sc->add_option("--muv",GLB::vector_mutation_rate,"genomic vector-wise mutation rate ["+std::to_string(GLB::vector_mutation_rate)+"]")->check(CLI::Bound(0.0,1.0));
    sc->add_option("--seed",GLB::seed,"set the seed")->check(CLI::Range(0,INT_MAX));
    sc->add_flag("--use-point-mu",GLB::use_point_mu,"enable point-wise mutations ["+std::to_string(GLB::use_point_mu)+"]");
    sc->add_flag("--use-col-mu",GLB::use_col_mu,"enable vector-wise col mutations ["+std::to_string(GLB::use_col_mu)+"]");
    sc->add_flag("--use-row-mu",GLB::use_row_mu,"enable vector-wise row mutations ["+std::to_string(GLB::use_row_mu)+"]");
    sc->add_flag("--local-mu",GLB::local_locality_mu,"enable local 'point-offset' mutations ["+std::to_string(GLB::local_locality_mu)+"]");
    sc->add_option("--save",GLB::savefilename,"file to save lod to [none]");
  }

  /* analyze subcommand */
  sc_analyze->add_option("filename",GLB::loadfilename,"lod file to load from")->check(CLI::ExistingFile);
  sc_analyze->add_option("--save",GLB::savefilename,"file to save lod to [none]");
  sc_analyze->add_option("--pick",GLB::gen_pick,"gens to analyze from lod [0,1,-1] [beg,skp,end]")->delimiter(':')->expected(3);
  sc_analyze->add_option("-n,--ntrials",GLB::ntrials,"times to mutate and get fitness per resolution ["+std::to_string(GLB::ntrials)+"]")->check(CLI::PositiveNumber);
  sc_analyze->add_option("-m,--maxmutations",GLB::maxmutations,"starting from 1, up to # of concurrent mutations to investigate ["+std::to_string(GLB::maxmutations)+"]")->check(CLI::PositiveNumber);
  sc_analyze->add_option("--rate",GLB::speed_change,"rate of environmental change ["+std::to_string(GLB::speed_change)+"]");
  sc_analyze->add_flag("-w,--fitness",GLB::show_fitness,"enable inspecting the fitness");
  sc_analyze->add_flag("-p,--phenotype",GLB::show_phenotype,"enable inspecting the phenotype array");
  sc_analyze->add_flag("-r,--robustness",GLB::show_robustness,"enable inspecting the robustness");

  /* info subcommand */
  sc_info->add_option("filename",GLB::loadfilename,"lod file to load from")->check(CLI::ExistingFile);

  CLI11_PARSE(app, argc, argv);

  /* available fitness functions */
  // ENV::MAXONES::evaluate
  // ENV::NKTREADMILL::evaluate

  if (sc_evolve->parsed()) {
    RUN::grn_nktreadmill_haploid_evolution(ENV::NKTREADMILL::evaluate);
  } else if (sc_test->parsed()) {
    RUN::grn_nktreadmill_haploid_evolution(ENV::MAXONES::evaluate);
  } else if (sc_analyze->parsed()) {
    RUN::analyze(ENV::NKTREADMILL::evaluate);
  } else if (sc_info->parsed()) {
    RUN::info();
  }

  return(0);
}

//int main() {
//  auto seed = getpid();
//  srand(seed); dsrand(seed);
//  Agent::configure({ .num_genes=GLB::G, .mu_point=GLB::point_mutation_rate/(GLB::G*GLB::G), .mu_vector=GLB::vector_mutation_rate/GLB::G });
//  std::vector<std::shared_ptr<Agent>> pop;
//  for (int i=0; i<100; i++) { pop.emplace_back(std::make_shared<Agent>(INIT::RANDOM)); }
//  for (int g=0; g<100; g++) {
//    auto agent = pop[g];
//    agent->initGRN();
//    std::ofstream outFS("states.csv",ios::trunc|ios::out);
//    std::ostream& out = outFS;
//    for (int i=0; i<800; i++) {
//      agent->runGRN(1);
//      agent->developPhenotypeFromGRN();
//      if (g > 97) {
//        for (int n=0; n<GLB::N-1; n++) {
//          out << agent->grn->state[n] << ",";
//          std::cout << agent->phenotype[n];
//        }
//        out << agent->grn->state[GLB::N-1] << std::endl;
//        std::cout << std::endl;
//      }
//    }
//  }
//  return(0);
//}

/* Arend's script-like main() */
//int main(int argc, char* argv[]) {
//  /* set up parameters */
//  using namespace GLB; // to easily set global variables below
//  // required
//  G=40;
//  N=20;
//  K=3;
//  // optional
//  generations_limit=10'000;
//  pop_size=100;
//  speed_change=0.001; // rate of environment change
//  developmental_updates = 200;
//  point_mutation_rate = 0.0005f;
//  local_locality_mu = false; // true = incremental genomic changes via mutation
//  savefilename="out.bin"; // non-empty saves
//
//  /* run evolution (& save data) */
//  // run evolution & write lod file with all genomes (5 million gen lod ~ 20 MB)
//  RUN::grn_nktreadmill_haploid_evolution(ENV::NKTREADMILL::evaluate);
//  /* currently available fitness functions */
//  // ENV::MAXONES::evaluate
//  // ENV::NKTREADMILL::evaluate
//
//  /* analyze subset of data (robustness, etc.) */
//  // do analysis on resulting lod file
//  loadfilename=savefilename;
//  savefilename="results.csv";
//  gen_pick={0,1'000,-1}; // [start,stride,end] -1 is "the final" if you don't know exactly
//  maxmutations=5; // try 1-mutations, 2-mutations.. n-mutations
//  ntrials=500; // how many trials to samples random n-mutations before averaging
//  // analyze robustness and write results to savefilename if non-empty string
//  RUN::analyze(ENV::NKTREADMILL::evaluate);
//}

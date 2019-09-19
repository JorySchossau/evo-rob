#include <chrono>
#include <cmath>
#include <unordered_map>

#ifndef PROFILING
struct Timing {
  static void Show();
  static void ShowRelativeTo(const std::string& /*key_of_total*/);
};
void Timing::Show() { }
void Timing::ShowRelativeTo(const std::string& key_of_total) { }
#define START(x)
#define PAUSE(x)
#define RESUME(x)
#define END(x)
#else
// custom profiling code
struct Timing {
  public:
    std::chrono::duration<float> elapsed_this_cycle {0.0f};
    std::chrono::duration<float> elapsed_total {0.0f};
    int cycles{0};
    static void Show();
    static void ShowRelativeTo(const std::string& /*key_of_total*/);
    static double round(const double& /*value*/, const int& /*places*/);
};
double Timing::round(const double& value, const int& places) {
  return std::round(value*pow(10,places))/pow(10,places);
}
std::unordered_map<std::string,Timing> times;
void Timing::Show() {
  for (auto const& [key, time] : times) {
    float elapsed_total = time.elapsed_total.count();
    float elapsed_this_cycle = time.elapsed_this_cycle.count();
    std::cout << key << " " << Timing::round(elapsed_this_cycle,4)*100 << "s over " << time.cycles << " iterations avg of " << Timing::round(elapsed_total/time.cycles,5) << "s per cycle." << std::endl;
  }
}
void Timing::ShowRelativeTo(const std::string& key_of_total) {
  if ( (times.find(key_of_total) == times.end()) || (times[key_of_total].elapsed_total.count() == 0)) {
    std::cout << "Error: No timing for section '"+key_of_total+"' been recorded." << std::endl;;
    std::cout << "       Do you have a 'START' and 'END' section in the code?" << std::endl;
    return;
  }
  float alltime = times[key_of_total].elapsed_total.count();
  for (auto const& [key, time] : times) {
    float elapsed_total = time.elapsed_total.count();
    std::cout << Timing::round(elapsed_total/alltime,4)*100 << "%)\t" << key << " " << Timing::round(elapsed_total,5) << "s over " << time.cycles << " iterations avg of " << Timing::round(elapsed_total/time.cycles,4) << "s per cycle" << std::endl;
  }
}
#define START(x) if (times.find(#x) == times.end()) times[#x] = {}; times[#x].timepoint = std::chrono::high_resolution_clock::now(); times[#x].elapsed_this_cycle = std::chrono::duration<float>::zero();

#define PAUSE(x) times[#x].elapsed_this_cycle += std::chrono::high_resolution_clock::now()-times[#x].timepoint;

#define RESUME(x) times[#x].timepoint = std::chrono::high_resolution_clock::now();

#define END(x) times[#x].elapsed_this_cycle += std::chrono::high_resolution_clock::now()-times[#x].timepoint; times[#x].cycles++; times[#x].elapsed_total += times[#x].elapsed_this_cycle;

#endif


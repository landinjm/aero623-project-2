#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <libassert/assert.hpp>
#include <string>
#include <unordered_map>

class Timer
{
public:
  Timer() = default;

  Timer(const Timer&) = delete;

  Timer& operator=(const Timer&) = delete;

  Timer(Timer&&) = delete;

  Timer& operator=(Timer&&) = delete;

  static Timer& instance()
  {
    static Timer timer;
    return timer;
  }

  void begin_section(const std::string& name)
  {
    auto& stats = sections[name];

    DEBUG_ASSERT(
      !stats.is_running,
      "Timer section " + name +
        " is already running. Make sure you call end_section() before "
        "calling begin_section().");

    stats.start_time = std::chrono::high_resolution_clock::now();
    stats.is_running = true;
  }

  void end_section(const std::string& name)
  {
    auto end_time = std::chrono::high_resolution_clock::now();

    DEBUG_ASSERT(
      sections.contains(name),
      "Timer section " + name +
        " does not contain an entry. Make sure you call end_section() before "
        "calling begin_section().");

    auto& stats = sections[name];
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - stats.start_time);
    stats.total_time += duration.count();
    stats.call_count++;
    stats.is_running = false;
  }

  void summary() const
  {
    // Print header
    std::cout << "\n" << std::endl;
    std::cout << std::left << std::setw(20) << "Section" << std::right
              << std::setw(15) << "Call Count" << std::setw(18)
              << "Total Time (ms)" << std::setw(18) << "Average (ms)" << "\n";
    std::cout << std::string(71, '-') << "\n";

    // Print data
    for (const auto& [name, stats] : sections) {
      double avg_time =
        stats.call_count > 0 ? stats.total_time / stats.call_count : 0.0;

      std::cout << std::left << std::setw(20) << name << std::right
                << std::setw(15) << stats.call_count << std::setw(18)
                << std::fixed << std::setprecision(6) << stats.total_time
                << std::setw(18) << std::fixed << std::setprecision(6)
                << avg_time << "\n";
    }

    // Print footer
    std::cout << std::string(71, '-') << "\n";
  }

private:
  struct SectionStats
  {
    unsigned int call_count = 0;
    double total_time = 0.0;
    std::chrono::high_resolution_clock::time_point start_time;
    bool is_running = false;
  };

  std::unordered_map<std::string, SectionStats> sections;
};

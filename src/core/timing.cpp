#include "timing.hpp"
#include <functional>
#include <string>

static void default_logger(const std::string& name, double ms) {
    (void)name; (void)ms; // no-op
}

ScopedTimer::~ScopedTimer() {
    using Clock = std::chrono::steady_clock;
    const auto end = Clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - start_).count();
    if (cb_) cb_(name_, ms);
}

double now_ms() {
    static const auto t0 = std::chrono::steady_clock::now();
    const auto t = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t - t0).count();
}

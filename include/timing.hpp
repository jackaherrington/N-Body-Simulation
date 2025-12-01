#pragma once

#include <chrono>
#include <string>
#include <functional>

// Simple scoped timer utility for measuring code regions.
// Usage:
//   {
//     ScopedTimer t("phase", logger);
//     ... work ...
//   } // logs elapsed
// The logger is a callable taking (const std::string& name, double ms)
class ScopedTimer {
public:
    template <typename Logger>
    ScopedTimer(const std::string& name, Logger&& logger)
    : name_(name), start_(Clock::now()), cb_([logger](const std::string& n, double ms){ logger(n, ms); }) {}

    ~ScopedTimer();
private:
    using Clock = std::chrono::steady_clock;
    std::string name_;
    Clock::time_point start_;
    std::function<void(const std::string&, double)> cb_{};
};

// Helper timing functions
double now_ms();

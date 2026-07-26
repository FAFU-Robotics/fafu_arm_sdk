#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include "hightorque_serial.hpp"

#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

class PseudoTerminal {
public:
    PseudoTerminal() {
        master_ = ::posix_openpt(O_RDWR | O_NOCTTY);
        if (master_ < 0 || ::grantpt(master_) != 0 ||
            ::unlockpt(master_) != 0) {
            throw std::runtime_error("failed to create pseudo terminal");
        }
        const char* name = ::ptsname(master_);
        if (name == nullptr) {
            throw std::runtime_error("failed to resolve pseudo terminal");
        }
        slave_ = name;
    }

    ~PseudoTerminal() {
        if (master_ >= 0) ::close(master_);
    }

    int master() const noexcept { return master_; }
    const std::string& slave() const noexcept { return slave_; }

private:
    int master_ = -1;
    std::string slave_;
};

class DebugBoardSimulator {
public:
    explicit DebugBoardSimulator(int master_fd)
        : master_fd_(master_fd), thread_(&DebugBoardSimulator::run, this) {}

    ~DebugBoardSimulator() { stop(); }

    void stop() {
        running_.store(false);
        if (thread_.joinable()) thread_.join();
    }

    std::vector<Clock::time_point> query_times(int motor_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = query_times_.find(motor_id);
        return it == query_times_.end()
            ? std::vector<Clock::time_point>{}
            : it->second;
    }

private:
    void handle_line(const std::string& raw_line) {
        std::string line = raw_line;
        if (!line.empty() && line.back() == '\r') line.pop_back();

        constexpr const char* prefix = "can send 800";
        if (line.rfind(prefix, 0) != 0 ||
            line.find(" 140400110F") == std::string::npos) {
            return;
        }
        const std::size_t id_pos = std::char_traits<char>::length(prefix);
        if (id_pos >= line.size() ||
            line[id_pos] < '1' || line[id_pos] > '9') {
            return;
        }
        const int motor_id = line[id_pos] - '0';
        {
            std::lock_guard<std::mutex> lock(mutex_);
            query_times_[motor_id].push_back(Clock::now());
        }

        // Deliberately drop motor 1. Motors 2 and 3 reply independently.
        if (motor_id == 1) return;
        const std::string reply_id = motor_id == 2 ? "200" : "300";
        const std::string response =
            "rcv " + reply_id +
            " 2404000A00000000000000210F00\r\n";
        const ssize_t written =
            ::write(master_fd_, response.data(), response.size());
        if (written != static_cast<ssize_t>(response.size())) {
            failed_.store(true);
        }
    }

    void run() {
        std::string buffer;
        while (running_.load()) {
            pollfd descriptor{};
            descriptor.fd = master_fd_;
            descriptor.events = POLLIN;
            const int ready = ::poll(&descriptor, 1, 20);
            if (ready < 0) {
                failed_.store(true);
                return;
            }
            if (ready == 0 || (descriptor.revents & POLLIN) == 0) continue;

            char chunk[512];
            const ssize_t count = ::read(master_fd_, chunk, sizeof(chunk));
            if (count <= 0) continue;
            buffer.append(chunk, static_cast<std::size_t>(count));

            while (true) {
                const std::size_t newline = buffer.find('\n');
                if (newline == std::string::npos) break;
                handle_line(buffer.substr(0, newline));
                buffer.erase(0, newline + 1);
            }
        }
    }

    int master_fd_;
    std::atomic<bool> running_{true};
    std::atomic<bool> failed_{false};
    mutable std::mutex mutex_;
    std::map<int, std::vector<Clock::time_point>> query_times_;
    std::thread thread_;
};

}  // namespace

int main() {
    try {
        PseudoTerminal terminal;
        DebugBoardSimulator simulator(terminal.master());
        hightorque::HightorqueSerial serial(terminal.slave(), 115200);

        constexpr double rate_hz = 20.0;
        serial.start_state_polling({1, 2, 3}, rate_hz);
        require(serial.is_async_rx(),
                "state polling must install its independent RX thread");

        const auto cache_deadline = Clock::now() + std::chrono::milliseconds(350);
        while (Clock::now() < cache_deadline) {
            if (serial.get_cached_state(2) && serial.get_cached_state(3)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        require(!serial.get_cached_state(1),
                "the deliberately dropped motor must remain absent");
        require(serial.get_cached_state(2).has_value() &&
                serial.get_cached_state(3).has_value(),
                "a dropped motor must not block later motor cache updates");

        const auto rounds_deadline = Clock::now() + std::chrono::milliseconds(500);
        while (Clock::now() < rounds_deadline &&
               simulator.query_times(1).size() < 6) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }

        serial.stop_state_polling();
        require(!serial.is_async_rx(),
                "stopping polling must stop the RX thread it auto-started");
        simulator.stop();

        const auto motor1 = simulator.query_times(1);
        const auto motor2 = simulator.query_times(2);
        const auto motor3 = simulator.query_times(3);
        require(!motor1.empty() && !motor2.empty() && !motor3.empty(),
                "the first polling round did not emit every query");
        require(std::chrono::duration_cast<std::chrono::milliseconds>(
                    motor3.front() - motor1.front()).count() < 100,
                "queries were serialized behind a per-motor reply timeout");
        require(motor1.size() >= 6,
                "absolute-deadline poller did not complete six rounds");

        const double observed_ms =
            std::chrono::duration<double, std::milli>(
                motor1[5] - motor1[0]).count();
        const double expected_ms = 5.0 * (1000.0 / rate_hz);
        require(std::abs(observed_ms - expected_ms) < 60.0,
                "polling deadlines drifted instead of tracking the absolute schedule");

        serial.close();
        std::cout << "Linux serial pipeline polling passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

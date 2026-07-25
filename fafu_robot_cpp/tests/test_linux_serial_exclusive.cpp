#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <serial/serial.h>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

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

    const std::string& slave() const noexcept { return slave_; }

private:
    int master_ = -1;
    std::string slave_;
};

}  // namespace

int main() {
    PseudoTerminal first_port;
    PseudoTerminal second_port;
    serial::Serial first(first_port.slave(), 115200);

    // Independent ports must remain usable at the same time.
    serial::Serial independent(second_port.slave(), 115200);
    if (!first.isOpen() || !independent.isOpen()) {
        std::cerr << "independent pseudo terminals did not open\n";
        return EXIT_FAILURE;
    }

    const pid_t child = ::fork();
    if (child < 0) {
        std::cerr << "fork failed\n";
        return EXIT_FAILURE;
    }
    if (child == 0) {
        try {
            serial::Serial duplicate(first_port.slave(), 115200);
            _exit(EXIT_FAILURE);
        } catch (const std::exception&) {
            _exit(EXIT_SUCCESS);
        }
    }

    int status = 0;
    if (::waitpid(child, &status, 0) != child ||
        !WIFEXITED(status) || WEXITSTATUS(status) != EXIT_SUCCESS) {
        std::cerr << "second process was able to open the locked port\n";
        return EXIT_FAILURE;
    }
    std::cout << "Linux serial exclusivity passed\n";
    return EXIT_SUCCESS;
}

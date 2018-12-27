#ifdef _WIN32

#include "Tests/WindowsTests.h"

#endif

int main(int argc, char **argv) {

#ifdef _WIN32

    auto windows = new WindowsTests();

    windows->run();

#endif

    return 0;
}

#include "tests/general/GeneralTests.h"

#ifdef _WIN32

#include "tests/windows/WindowsTests.h"

#endif

int main(int argc, char **argv) {

#ifdef _WIN32

    auto windows = new WindowsTests();

    windows->run();

#endif

    auto general = new GeneralTests();

    general->run();

    return 0;
}

#include <iostream>

#include "Tracker.h"

extern "C" {
    const char* PrintHello (){
        return "Hello";
    };

}


int main() {

    Tracker tracker;

    tracker.computerCameraPose(nullptr, nullptr, 0, 0);

    return 0;
}
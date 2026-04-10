#include <Arduino.h>

bool getPauseState();

void setupBLE(const char* deviceName);

void notifyCountIncremented(uint32_t count);
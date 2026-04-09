#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <ble_server.h>

int i = 0;

Adafruit_MPU6050 mpu;

void setup(void) {
  Serial.begin(9600);
  while (!Serial)
    delay(10); 

  setupBLE("Smart_Hook");

  delay(100);
}

void loop() {
  Serial.println("increment count");
  notifyCountIncremented(i);
  i++;
  
  delay(10000);
}
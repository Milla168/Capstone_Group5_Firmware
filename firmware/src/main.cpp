// #include <Arduino.h>
// #include <Wire.h>
// #include <Adafruit_Sensor.h>
// #include <Adafruit_BNO055.h>

// Adafruit_BNO055 bno = Adafruit_BNO055(55);

// const int SAMPLE_RATE_HZ = 100;
// const unsigned long SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ;

// unsigned long lastSampleTime = 0;
// unsigned long startTime = 0;

// void setup(void)
// {
//   Serial.begin(115200);
//   delay(2000);

//   Serial.println("==============================");
//   Serial.println("BNO055 Motion Data Logger");
//   Serial.println("==============================");

//   Serial.print("[LOG] Initializing BNO055... ");

//   if (!bno.begin())
//   {
//     Serial.println("[ERROR] Failed to find BNO055");
//     while (1) delay(10);
//   }

//   Serial.println("OK");

//   delay(1000);

//   bno.setExtCrystalUse(true);

//   Serial.println("[LOG] BNO055 Ready");
//   Serial.println("==============================");

//   bno.setMode(OPERATION_MODE_IMUPLUS);

//   startTime = millis();
// }

// void loop()
// {
  
//   if (millis() - lastSampleTime >= SAMPLE_INTERVAL_MS)
//   {
//     lastSampleTime = millis();

//     // Raw accelerometer
//     imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);

//     // Raw gyroscope
//     imu::Vector<3> gyro = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);

//     float ax = accel.x() / 9.81;
//     float ay = accel.y() / 9.81;
//     float az = accel.z() / 9.81;

//     float gx = gyro.x();
//     float gy = gyro.y();
//     float gz = gyro.z();

//     float accel_mag = sqrt(ax*ax + ay*ay + az*az);
//     float gyro_mag  = sqrt(gx*gx + gy*gy + gz*gz);

//     unsigned long timestamp = millis() - startTime;

//     Serial.print(timestamp);
//     Serial.print(",");

//     Serial.print(ax,4);
//     Serial.print(",");
//     Serial.print(ay,4);
//     Serial.print(",");
//     Serial.print(az,4);
//     Serial.print(",");

//     Serial.print(gx,2);
//     Serial.print(",");
//     Serial.print(gy,2);
//     Serial.print(",");
//     Serial.print(gz,2);
//     Serial.print(",");

//     Serial.print(accel_mag,4);
//     Serial.print(",");
//     Serial.println(gyro_mag,2);
//   }
// }




// ========================================================================================


#include <Arduino.h>
#include <Adafruit_ISM330DHCX.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_LSM6DSOX lsmdsox;

lsm6ds_accel_range_t _ACCEL_RANGE = LSM6DS_ACCEL_RANGE_2_G;
lsm6ds_gyro_range_t  _GYRO_RANGE  = LSM6DS_GYRO_RANGE_250_DPS;
lsm6ds_data_rate_t   _DATA_RATE   = LSM6DS_RATE_104_HZ;

const int SAMPLE_RATE_HZ = 100;
const unsigned long SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ;
unsigned long lastSampleTime = 0;
unsigned long startTime = 0;

// Function prototype
void print_sensor_settings();

void setup(void) {
  Serial.begin(115200);

  while (!Serial) delay(10);

  Serial.println("==============================");
  Serial.println("Crochet Motion Data Logger");
  Serial.println("==============================");

  Serial.print("[LOG] Initializing LSM6DSOX... ");

  if (!lsmdsox.begin_I2C()) {
    Serial.println("[ERROR] Failed to find LSM6DSOX chip");
    while (1) delay(10);
  }
  Serial.println("OK");

  Serial.print("[LOG] Configuring LSM6DSOX... ");
  lsmdsox.setAccelRange(_ACCEL_RANGE);
  lsmdsox.setGyroRange(_GYRO_RANGE);
  lsmdsox.setAccelDataRate(_DATA_RATE);
  lsmdsox.setGyroDataRate(_DATA_RATE);
  Serial.println("OK");

  Serial.println("[LOG] LSM6DSOX Settings:");
  print_sensor_settings();

  Serial.println("[LOG] LSM6DSOX Ready");
  Serial.println("==============================");

  delay(1000);

  startTime = millis();
}

void loop() {
  if (millis() - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = millis();

    sensors_event_t accel, gyro, temp;
    lsmdsox.getEvent(&accel, &gyro, &temp);

    // Accelerometer (m/s² to g)
    float ax = accel.acceleration.x / 9.81;
    float ay = accel.acceleration.y / 9.81;
    float az = accel.acceleration.z / 9.81;

    // Gyroscope (rad/s to °/s)
    float gx = gyro.gyro.x * 57.2958;
    float gy = gyro.gyro.y * 57.2958;
    float gz = gyro.gyro.z * 57.2958;

    // Calculate magnitudes
    float accel_mag = sqrt(ax*ax + ay*ay + az*az);
    float gyro_mag  = sqrt(gx*gx + gy*gy + gz*gz);

    unsigned long timestamp = millis() - startTime;

    // debugPrint(timestamp, ax, ay, az, gx, gy, gz, accel_mag, gyro_mag);

    // Output CSV line
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(ax, 4);
    Serial.print(",");
    Serial.print(ay, 4);
    Serial.print(",");
    Serial.print(az, 4);
    Serial.print(",");
    Serial.print(gx, 2);
    Serial.print(",");
    Serial.print(gy, 2);
    Serial.print(",");
    Serial.print(gz, 2);
    Serial.print(",");
    Serial.print(accel_mag, 4);
    Serial.print(",");
    Serial.println(gyro_mag, 2);
  }
}

void print_sensor_settings() {
  Serial.print("  Accelerometer range: ");
  switch (lsmdsox.getAccelRange()) {
    case LSM6DS_ACCEL_RANGE_2_G:  Serial.println("±2G");  break;
    case LSM6DS_ACCEL_RANGE_4_G:  Serial.println("±4G");  break;
    case LSM6DS_ACCEL_RANGE_8_G:  Serial.println("±8G");  break;
    case LSM6DS_ACCEL_RANGE_16_G: Serial.println("±16G"); break;
  }

  Serial.print("  Gyroscope range:     ");
  switch (lsmdsox.getGyroRange()) {
    case LSM6DS_GYRO_RANGE_125_DPS:  Serial.println("±125°/s");  break;
    case LSM6DS_GYRO_RANGE_250_DPS:  Serial.println("±250°/s");  break;
    case LSM6DS_GYRO_RANGE_500_DPS:  Serial.println("±500°/s");  break;
    case LSM6DS_GYRO_RANGE_1000_DPS: Serial.println("±1000°/s"); break;
    case LSM6DS_GYRO_RANGE_2000_DPS: Serial.println("±2000°/s"); break;
    case ISM330DHCX_GYRO_RANGE_4000_DPS: Serial.println("±4000°/s"); break;
  }

  Serial.print("  Accel data rate:     ");
  switch (lsmdsox.getAccelDataRate()) {
    case LSM6DS_RATE_SHUTDOWN:  Serial.println("Shutdown");    break;
    case LSM6DS_RATE_12_5_HZ:  Serial.println("12.5 Hz");     break;
    case LSM6DS_RATE_26_HZ:    Serial.println("26 Hz");       break;
    case LSM6DS_RATE_52_HZ:    Serial.println("52 Hz");       break;
    case LSM6DS_RATE_104_HZ:   Serial.println("104 Hz");      break;
    case LSM6DS_RATE_208_HZ:   Serial.println("208 Hz");      break;
    case LSM6DS_RATE_416_HZ:   Serial.println("416 Hz");      break;
    case LSM6DS_RATE_833_HZ:   Serial.println("833 Hz");      break;
    case LSM6DS_RATE_1_66K_HZ: Serial.println("1.66 KHz");    break;
    case LSM6DS_RATE_3_33K_HZ: Serial.println("3.33 KHz");    break;
    case LSM6DS_RATE_6_66K_HZ: Serial.println("6.66 KHz");    break;
  }

  Serial.print("  Gyro data rate:      ");
  switch (lsmdsox.getGyroDataRate()) {
    case LSM6DS_RATE_SHUTDOWN:  Serial.println("Shutdown");    break;
    case LSM6DS_RATE_12_5_HZ:  Serial.println("12.5 Hz");     break;
    case LSM6DS_RATE_26_HZ:    Serial.println("26 Hz");       break;
    case LSM6DS_RATE_52_HZ:    Serial.println("52 Hz");       break;
    case LSM6DS_RATE_104_HZ:   Serial.println("104 Hz");      break;
    case LSM6DS_RATE_208_HZ:   Serial.println("208 Hz");      break;
    case LSM6DS_RATE_416_HZ:   Serial.println("416 Hz");      break;
    case LSM6DS_RATE_833_HZ:   Serial.println("833 Hz");      break;
    case LSM6DS_RATE_1_66K_HZ: Serial.println("1.66 KHz");    break;
    case LSM6DS_RATE_3_33K_HZ: Serial.println("3.33 KHz");    break;
    case LSM6DS_RATE_6_66K_HZ: Serial.println("6.66 KHz");    break;
  }
}


// void debugPrint(
//   unsigned long timestamp,
//   float ax, float ay, float az,
//   float gx, float gy, float gz,
//   float accel_mag, float gyro_mag
// ) {
//   Serial.println("----- SENSOR DEBUG -----");

//   Serial.print("Time (ms): ");
//   Serial.println(timestamp);

//   Serial.println("Accelerometer (g):");
//   Serial.print("  X: "); Serial.println(ax, 4);
//   Serial.print("  Y: "); Serial.println(ay, 4);
//   Serial.print("  Z: "); Serial.println(az, 4);

//   Serial.println("Gyroscope (deg/s):");
//   Serial.print("  X: "); Serial.println(gx, 2);
//   Serial.print("  Y: "); Serial.println(gy, 2);
//   Serial.print("  Z: "); Serial.println(gz, 2);

//   Serial.print("Accel Magnitude: ");
//   Serial.println(accel_mag, 4);

//   Serial.print("Gyro Magnitude: ");
//   Serial.println(gyro_mag, 2);

//   Serial.println("------------------------");
//   Serial.println();
// }



// ========================================================================================


// #include <Arduino.h>
// #include <Adafruit_MPU6050.h>
// #include <Adafruit_Sensor.h>
// #include <Wire.h>

// Adafruit_MPU6050 mpu;

// mpu6050_accel_range_t _ACCEL_RANGE = MPU6050_RANGE_2_G;
// mpu6050_gyro_range_t _GYRO_RANGE = MPU6050_RANGE_250_DEG;
// mpu6050_bandwidth_t _BANDWIDTH = MPU6050_BAND_21_HZ;

// // ADD THESE MISSING VARIABLES:
// const int SAMPLE_RATE_HZ = 100;
// const unsigned long SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ;  // = 10ms
// unsigned long lastSampleTime = 0;
// unsigned long startTime = 0;

// // Function prototype
// void print_sensor_settings();

// void setup(void) {
//   Serial.begin(115200);

//   while (!Serial) delay(10);

//   Serial.println("==============================");
//   Serial.println("Crochet Motion Data Logger");
//   Serial.println("==============================");

//   Serial.print("[LOG] Initializing MPU6050... ");

//   if (!mpu.begin()) {
//     Serial.println("[ERROR] Failed to find MPU6050 chip");
//     while (1) delay(10);
//   }
//   Serial.println("OK");

//   Serial.print("[LOG] Configuring MPU6050... ");
//   mpu.setAccelerometerRange(_ACCEL_RANGE);
//   mpu.setGyroRange(_GYRO_RANGE);
//   mpu.setFilterBandwidth(_BANDWIDTH);
//   Serial.println("OK");

//   Serial.println("[LOG] MPU6050 Settings:");
//   print_sensor_settings();

//   Serial.println("[LOG] MPU6050 Ready");
//   Serial.println("==============================");

//   delay(1000);
  
//   startTime = millis();
// }

// void loop() {
//   if (millis() - lastSampleTime >= SAMPLE_INTERVAL_MS) {
//     lastSampleTime = millis();
    
//     // Get sensor events
//     sensors_event_t accel, gyro, temp;
//     mpu.getEvent(&accel, &gyro, &temp);
    
//     // Accelerometer (m/s² to g)
//     float ax = accel.acceleration.x / 9.81;
//     float ay = accel.acceleration.y / 9.81;
//     float az = accel.acceleration.z / 9.81;
    
//     // Gyroscope (rad/s to °/s)
//     float gx = gyro.gyro.x * 57.2958;
//     float gy = gyro.gyro.y * 57.2958;
//     float gz = gyro.gyro.z * 57.2958;
    
//     // Calculate magnitudes
//     float accel_mag = sqrt(ax*ax + ay*ay + az*az);
//     float gyro_mag = sqrt(gx*gx + gy*gy + gz*gz);
    
//     unsigned long timestamp = millis() - startTime;
    
//     // Output CSV line
//     Serial.print(timestamp);
//     Serial.print(",");
//     Serial.print(ax, 4);
//     Serial.print(",");
//     Serial.print(ay, 4);
//     Serial.print(",");
//     Serial.print(az, 4);
//     Serial.print(",");
//     Serial.print(gx, 2);
//     Serial.print(",");
//     Serial.print(gy, 2);
//     Serial.print(",");
//     Serial.print(gz, 2);
//     Serial.print(",");
//     Serial.print(accel_mag, 4);
//     Serial.print(",");
//     Serial.println(gyro_mag, 2);
//   }
// }

// void print_sensor_settings() {
//   Serial.print("       Accelerometer range: ");
//   switch (mpu.getAccelerometerRange()) {
//     case MPU6050_RANGE_2_G:  Serial.println("±2G");  break;
//     case MPU6050_RANGE_4_G:  Serial.println("±4G");  break;
//     case MPU6050_RANGE_8_G:  Serial.println("±8G");  break;
//     case MPU6050_RANGE_16_G: Serial.println("±16G"); break;
//   }
  
//   Serial.print("       Gyroscope range: ");
//   switch (mpu.getGyroRange()) {
//     case MPU6050_RANGE_250_DEG:  Serial.println("±250°/s");  break;
//     case MPU6050_RANGE_500_DEG:  Serial.println("±500°/s");  break;
//     case MPU6050_RANGE_1000_DEG: Serial.println("±1000°/s"); break;
//     case MPU6050_RANGE_2000_DEG: Serial.println("±2000°/s"); break;
//   }
  
//   Serial.print("       Filter bandwidth: ");
//   switch (mpu.getFilterBandwidth()) {
//     case MPU6050_BAND_260_HZ: Serial.println("260 Hz"); break;
//     case MPU6050_BAND_184_HZ: Serial.println("184 Hz"); break;
//     case MPU6050_BAND_94_HZ:  Serial.println("94 Hz");  break;
//     case MPU6050_BAND_44_HZ:  Serial.println("44 Hz");  break;
//     case MPU6050_BAND_21_HZ:  Serial.println("21 Hz");  break;
//     case MPU6050_BAND_10_HZ:  Serial.println("10 Hz");  break;
//     case MPU6050_BAND_5_HZ:   Serial.println("5 Hz");   break;
//   }
// }
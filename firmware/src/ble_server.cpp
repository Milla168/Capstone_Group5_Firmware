#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <ble_server.h>

// Service and charactersitic UUIDs
#define SERVICE_UUID "3fef7f66-b8a3-4c26-ba26-4aa6442d980b"
#define COUNTER_CHARACTERISTIC_UUID "2807becc-e2e0-464b-aaf4-e6752ec49e79"

BLECharacteristic* pCounterCharacteristic = NULL;
bool deviceConnected = false;
bool isPaused = false;

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
    };

    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      BLEDevice::startAdvertising(); //restart advertising so app can reconnect if link drops
    }
};

//NOTE: on write/notify, the counter characteristic is the stitch count
// BUT on read, the counter characteristic is the pause flag
class MyCharacteristicCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic* pCharacteristic){
        std::string value = pCharacteristic->getValue();

        // Reconstruct the uint32_t from the 4 bytes (Little Endian)
        uint8_t flag = (uint8_t)value[0];
        Serial.print("FLAG");
        Serial.print(flag);
        isPaused = (flag == 1);

        Serial.print("BLE Command: ");
        Serial.println(isPaused ? "PAUSE" : "RESUME");           
        
    }
};

bool getPauseState(){
    return isPaused;
}

void setupBLE(const char* deviceName = "Smart_Hook"){
    BLEDevice::init(deviceName);
    BLEServer *pServer = BLEDevice::createServer();
    BLEService *pService = pServer->createService(SERVICE_UUID);
    pServer->setCallbacks(new MyServerCallbacks()); //connect callbacks
    pCounterCharacteristic = pService->createCharacteristic(
                        COUNTER_CHARACTERISTIC_UUID,
                        BLECharacteristic::PROPERTY_READ | 
                        BLECharacteristic::PROPERTY_WRITE |
                        BLECharacteristic::PROPERTY_NOTIFY // enables notify()
                    );
    BLEDescriptor *pDescriptor = new BLEDescriptor((uint16_t)0x2902); //for notifications
    pCounterCharacteristic->addDescriptor(pDescriptor);
    pCounterCharacteristic->setCallbacks(new MyCharacteristicCallbacks());

    pService->start();

    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(0x06);  // functions that help with iPhone connections issue
    pAdvertising->setMinPreferred(0x12);
    BLEDevice::startAdvertising();
    Serial.println("BLE advertising started...");
}

void notifyCountIncremented(uint32_t count){
    if(deviceConnected){
        pCounterCharacteristic->setValue((uint8_t*)&count, 4); // converts count to uint32_t byte array
        pCounterCharacteristic->notify();
    }
}

/* very generally, how usage main should look:
//globally defined count variable: e.g. uint32_t count = 0

setupBLE()

// create task to read from IMU and run ML model to determine stitch
// once the ML model determines stitch or no stitch: 
count++
notifyCountIncremented(count)
etc.
*/
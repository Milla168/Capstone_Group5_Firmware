const int trigPin = 9;
const int echoPin = 10;
const int vibPin  = 5;

const int distanceThreshold = 10; // cm

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(vibPin, OUTPUT);

  digitalWrite(trigPin, LOW);
  digitalWrite(vibPin, LOW);

  Serial.begin(9600);
}

void loop() {
  long duration;
  float distance;

  // Send ultrasonic pulse
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Read echo time
  duration = pulseIn(echoPin, HIGH);

  // Convert to distance in cm
  distance = duration * 0.0343 / 2.0;

  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  // Vibrate if object is close
  if (distance > 0 && distance <= distanceThreshold) {
    analogWrite(vibPin, 80);
    delay(500);                  // vibrate for half a second
    digitalWrite(vibPin, LOW);
    delay(200);                  // small pause
  } else {
    digitalWrite(vibPin, LOW);
  }

  delay(100);
}
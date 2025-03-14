// Particle Systems for Rain and Splashes
ArrayList<Raindrop> raindrops;
ArrayList<Splash> splashes;
ArrayList<Obstacle> obstacles;
ArrayList<Drip> drips;           // New ArrayList for drip particles


// Adjustable Parameters
float gravity = 0.25;
float windForce = 0.4; //horizontal push
float rainSize = 5;
float rainSpeed = 13;
float splashIntensity = 4;//controls number of splash particles generated per splash
float rainDensity = 1; //controls the number of raindrops per frame
float splashInteractionRadius = 10;
float splashReproductionRate = 0.5;
boolean showUI = true;

// Toggle obstacles collision
boolean obstaclesEnabled = true;

//Draw the canvas and initialize the two arrays
void setup() {
  size(800, 600);
  raindrops = new ArrayList<Raindrop>();
  splashes = new ArrayList<Splash>();
  obstacles = new ArrayList<Obstacle>();
  drips = new ArrayList<Drip>();      // Initialize drips


}
//main loop
void draw() {
  background(16,34,90); //blue background
  
  // Display each obstacle
  for (Obstacle obs : obstacles) {
    obs.display();
  }

  // For each obstacle, occasionally spawn a drip from its bottom edge.
  if (obstaclesEnabled) {
    for (Obstacle obs : obstacles) {
      // Adjust the probability to control drip frequency
      if (random(1) < 0.02) {
        // Choose a random x coordinate along the bottom of the obstacle
        float dripX = random(obs.x, obs.x + obs.w);
        float dripY = obs.y + obs.h; // bottom edge of the obstacle
        drips.add(new Drip(dripX, dripY));
      }
    }
  }
  
  // Generate new raindrops
  for (int i = 0; i < rainDensity; i++) {
    if (random(1) < 0.6) { //60% chance to spawn a raindrop per iteration
      raindrops.add(new Raindrop());
    }
  }
  
  // iterate over the entire list of raindrops to update and display them
  for (int i = raindrops.size() - 1; i >= 0; i--) {
    Raindrop drop = raindrops.get(i);
    drop.update();
    drop.display();
    
    // Compute the bottom of the raindrop (its drawn tail)
    PVector dropBottom = new PVector(
      drop.position.x + drop.velocity.x * (drop.length / rainSpeed),
      drop.position.y + drop.velocity.y * (drop.length / rainSpeed)
    );

    // Check collision with obstacles if enabled
    boolean collidedObstacle = false;
    float splashY = height; // default: ground level
    if (obstaclesEnabled) {
      for (Obstacle obs : obstacles) {
        // Check if the bottom of the raindrop has reached the top edge of the obstacle
        if (dropBottom.x >= obs.x && dropBottom.x <= obs.x + obs.w &&
            dropBottom.y >= obs.y) {
          collidedObstacle = true;
          splashY = obs.y; // splash exactly at the obstacle's top boundary
          break; // collision found; no need to check other obstacles
        }
      }
    }

    // If the drop has reached the ground or collided with an obstacle, create splashes.
    if (drop.position.y > height || collidedObstacle) {
      for (int j = 0; j < splashIntensity; j++) {
        splashes.add(new Splash(drop.position.x, splashY, 1));
      }
      raindrops.remove(i);
    } else if (drop.position.x < -50 || drop.position.x > width + 50) {
      raindrops.remove(i);
    }
  }
  
  // Handle splash interactions and updates
  handleSplashInteractions();
  
  // iterate over the entire list of splashes to update and display them
  for (int i = splashes.size() - 1; i >= 0; i--) {
    Splash splash = splashes.get(i);
    splash.update();
    splash.display();
    
    //if lifetime ended
    if (splash.isDead()) {
      splashes.remove(i);
    }
  }
  
  // Update and display drips
  for (int i = drips.size() - 1; i >= 0; i--) {
    Drip drip = drips.get(i);
    drip.update();
    drip.display();
    
    if (drip.position.y > height) {
      drips.remove(i);
    }
  }
  
  if (showUI) {
    displayUI();
  }
}

void handleSplashInteractions() {
  ArrayList<Splash> newSplashes = new ArrayList<Splash>();
  
  for (int i = 0; i < splashes.size(); i++) {
    
    Splash splash1 = splashes.get(i);
    if (splash1.isReproducing()) { //both splashes must be in isReproducing() state
      for (int j = i + 1; j < splashes.size(); j++) {
        Splash splash2 = splashes.get(j);
        
        if (splash2.isReproducing()) {
          float distance = PVector.dist(splash1.position, splash2.position);
          
          if (distance < splashInteractionRadius) {
            if (random(1) < splashReproductionRate) {
              PVector midPoint = PVector.add(splash1.position, splash2.position).div(2);
              
              // Create multiple secondary splashes for better visibility
              int newGeneration = min(splash1.generation, splash2.generation) + 1;
              if (newGeneration <= 3) {
                for (int k = 0; k < 3; k++) { // Create 3 new splashes per interaction
                  newSplashes.add(new Splash(midPoint.x, midPoint.y, newGeneration));
                }
              }
            }
            splash1.hasReproduced = true;
            splash2.hasReproduced = true;
          }
        }
      }
    }
  }
  
  splashes.addAll(newSplashes);
}

class Raindrop {
  PVector position;//current particle location
  PVector velocity;//vector for movement  
  float size; //raindrop thickness(horizontal)
  float length;
  
  Raindrop() {
    position = new PVector(random(-50, width + 50), random(-50, -10));
    float speedVariation = random(-1, 1);
    velocity = new PVector(0, rainSpeed + speedVariation);
    size = randomGaussian() * 0.5 + rainSize;
    size = constrain(size, 1, rainSize * 1.5);
    length = map(velocity.y, 0, rainSpeed + 1, 10, 20); // faster falling raindrops have longer streaks
  }
  
  void update() {
    velocity.y += gravity + random(-0.1, 0.1); 
    velocity.x = windForce + random(-0.05, 0.05);
    position.add(velocity);
  }
  
  void display() {
  pushMatrix();
  translate(position.x, position.y);
  // Compute the angle of the velocity so the ellipse aligns with the drop's direction
  float angle = atan2(velocity.y, velocity.x) - PI/2;
  rotate(angle);
  noStroke();
  fill(150, 200, 255, 200);
  // Draw an ellipse with a smaller width and larger height for a drop shape.
  ellipse(0, 0, size, size * 2);
  popMatrix();
}
}

class Splash {
  PVector position;
  PVector velocity;
  float lifetime;
  float maxLifetime;
  float size;
  float alpha;
  int generation;
  boolean hasReproduced; //Flags if this splash has already been used in an interaction.
  float energy;
  
  Splash(float x, float y, int gen) {
    position = new PVector(x, y);
    generation = gen;
    
    // Adjust properties based on generation
    float baseSpeed;
    switch(generation) {
      case 1:
        baseSpeed = random(2, 5);
        maxLifetime = random(25, 35);
        size = random(2, 4);
        break;
      case 2:
        baseSpeed = random(3, 6);  // Faster secondary splashes
        maxLifetime = random(30, 40);  // Longer lifetime
        size = random(3, 5);  // Larger size
        break;
      case 3:
        baseSpeed = random(4, 7);  // Even faster tertiary splashes
        maxLifetime = random(20, 30);
        size = random(2, 4);
        break;
      default:
        baseSpeed = random(2, 4);
        maxLifetime = random(20, 30);
        size = random(1, 3);
    }
    
    //compute splash direction
    float angle = random(-PI, 0);
    velocity = new PVector(cos(angle) * baseSpeed, sin(angle) * baseSpeed);
    
    //flash starts fully opaque(alpha 255)(no transparency)
    lifetime = maxLifetime;
    alpha = 255;
    hasReproduced = false;
    energy = 1.0;
  }
  
  void update() {
    velocity.y += gravity * 0.5;
    position.add(velocity);
    lifetime--;
    energy = lifetime / maxLifetime; 
    alpha = map(lifetime, 0, maxLifetime, 0, 255); //is remapped so that the splash fades out as its lifetime decreases.
  }
  
  void display() {
    strokeWeight(size);
    
    // Different colors and effects based on generation
    switch(generation) {
      case 1:
        // First generation: Dark blue splash
        stroke(45, 0, 252, alpha);
        point(position.x, position.y);
        break;
      case 2:
        // Second generation: Bright light-blue splash with trail
        stroke(4, 139, 203, alpha);
        point(position.x, position.y);
        // Add a subtle trail
        stroke(230, 200, 255, alpha * 0.5);
        line(position.x, position.y, 
             position.x - velocity.x * 2, 
             position.y - velocity.y * 2);
        break;
      case 3:
        // Third generation: Cyan splash with glow effect
        stroke(5, 255, 240, alpha);
        point(position.x, position.y);
        // Add glow effect
        strokeWeight(size * 2);
        stroke(150, 255, 255, alpha * 0.3);
        point(position.x, position.y);
        break;
    }
  }
  
  boolean isDead() {
    return lifetime <= 0;
  }
  
  //can reproduce if hasnt reproduced yet and still sufficient lifetime is left 
  boolean isReproducing() {
    return !hasReproduced && lifetime > maxLifetime * 0.6;
  }
}

// New Obstacle class (simple rectangle)
class Obstacle {
  float x, y, w, h;
  
  Obstacle(float x, float y, float w, float h) {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
  }
  
  void display() {
    fill(100, 100, 100);
    noStroke();
    rect(x, y, w, h);
  }
  
  // Check if a point is inside this obstacle (if needed elsewhere)
  boolean contains(PVector point) {
    return (point.x >= x && point.x <= x + w &&
            point.y >= y && point.y <= y + h);
  }
}

class Drip {
  PVector position;
  PVector velocity;
  float length; // Length of the drip trail
  
  Drip(float x, float y) {
    position = new PVector(x, y);
    velocity = new PVector(0, random(3, 5)); // initial downward speed
    length = random(7, 14);
  }
  
  void update() {
    velocity.y += gravity; // a gentle acceleration
    position.add(velocity);
  }
  
  void display() {
    stroke(150, 200, 255, 200);
    strokeWeight(3);
    // Draw a vertical line representing the drip
    line(position.x, position.y, position.x, position.y + length);
  }
}

// Create an obstacle where the user clicks
void mousePressed() {
  // Create a new obstacle centered on the mouse position.
  float obsW = 100; // obstacle width
  float obsH = 20;  // obstacle height
  float x = mouseX - obsW / 2;
  float y = mouseY - obsH / 2;
  obstacles.add(new Obstacle(x, y, obsW, obsH));
}

void displayUI() {
  fill(0, 150);
  rect(10, 10, 200, 220);
  
  fill(255);
  textSize(12);
  text("Controls:", 20, 30);
  text("UP/DOWN - Rain Speed: " + nf(rainSpeed, 0, 1), 20, 50);
  text("LEFT/RIGHT - Wind: " + nf(windForce, 0, 2), 20, 70);
  text("W/S - Rain Size: " + nf(rainSize, 0, 1), 20, 90);
  text("A/D - Splash Intensity: " + int(splashIntensity), 20, 110);
  text("Z/X - Rain Density: " + nf(rainDensity, 0, 2), 20, 130);
  text("Q/E - Splash Interaction: " + nf(splashInteractionRadius, 0, 1), 20, 150);
  text("Click to add an Obstacle", 20, 170);
  text("Press r to remove all obstacles", 20, 190);
  text("H - Toggle UI", 20, 210);
}

void keyPressed() {
  if (key == CODED) {
    if (keyCode == UP) rainSpeed = constrain(rainSpeed + 0.5, 1, 20);
    if (keyCode == DOWN) rainSpeed = constrain(rainSpeed - 0.5, 1, 15);
    if (keyCode == LEFT) windForce = constrain(windForce - 0.1, -2, 2);
    if (keyCode == RIGHT) windForce = constrain(windForce + 0.1, -2, 2);
  }
  
  if (key == 'w' || key == 'W') rainSize = constrain(rainSize + 0.5, 1, 10);
  if (key == 's' || key == 'S') rainSize = constrain(rainSize - 0.5, 1, 10);
  if (key == 'a' || key == 'A') splashIntensity = constrain(splashIntensity - 1, 1, 10);
  if (key == 'd' || key == 'D') splashIntensity = constrain(splashIntensity + 1, 1, 10);
  if (key == 'z' || key == 'Z') rainDensity = constrain(rainDensity - 1, 1, 4);
  if (key == 'x' || key == 'X') rainDensity = constrain(rainDensity + 1, 1, 4);
  if (key == 'q' || key == 'Q') splashInteractionRadius = constrain(splashInteractionRadius - 1, 5, 30);
  if (key == 'e' || key == 'E') splashInteractionRadius = constrain(splashInteractionRadius + 1, 5, 30);
  if (key == 'h' || key == 'H') showUI = !showUI;
  if (key == 'r' || key == 'R') {
    obstacles.clear();
  }
}

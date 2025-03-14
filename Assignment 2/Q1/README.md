# Rain & Splash Simulation with Obstacles, Drips, and Multi-Generation Splashes

This Processing sketch simulates a realistic rain environment with interactive features such as obstacles, water drips, and multi-generation splash interactions. It uses particle systems to create a dynamic and visually engaging simulation.

## Overview

The sketch features:
- **Raindrops:** Particles that fall from the top of the canvas with adjustable speed, size, and wind influence.
- **Splashes:** When a raindrop hits the ground or an obstacle, it generates splash particles. These splashes interact with each other to produce secondary and tertiary generations, enhancing the visual effect.
- **Obstacles:** Click anywhere on the canvas to create an obstacle. Raindrops that hit the top edge of an obstacle generate splashes at that boundary.
- **Drips:** Obstacles occasionally drip water from their bottom edge, adding to the realistic ambiance.
- **Interactive UI:** A heads-up display shows the current settings and available controls.

## Features

- **Rain Simulation:**  
  Adjust the rain density, speed, size, and wind force to create various weather effects.
  
- **Splash Interactions:**  
  Splashes are generated upon impact and interact with each other to create up to three generations of splash effects. Use parameters like splash intensity, interaction radius, and reproduction rate to control the effect.
  
- **Dynamic Obstacles:**  
  Obstacles can be added by mouse clicks. They influence the behavior of raindrops (which generate splashes when hitting the obstacle's top) and periodically drip water.
  
- **Water Drips:**  
  Drip particles are spawned randomly from the bottom edges of obstacles and fall with gravity, simulating water dripping down.
  
- **Interactive Controls:**  
  Use keyboard keys to adjust simulation parameters in real time, toggle the UI display, and clear obstacles.

## How to Run

1. **Install Processing:**  
   Download and install Processing from [processing.org](https://processing.org/).

2. **Open the Sketch:**  
   Open the provided `.pde` file in the Processing IDE.

3. **Run the Sketch:**  
   Click the **Run** button to start the simulation.

## Controls

- **Rain Parameters:**
  - **UP/DOWN Arrows:** Increase/Decrease Rain Speed.
  - **W/S Keys:** Increase/Decrease Rain Size.
  - **LEFT/RIGHT Arrows:** Decrease/Increase Wind Force.
  - **Z/X Keys:** Decrease/Increase Rain Density.
  - **A/D Keys:** Decrease/Increase Splash Intensity.

- **Splash Interaction:**
  - **Q/E Keys:** Decrease/Increase Splash Interaction Radius.

- **Obstacles & Drips:**
  - **Mouse Click:** Create a new obstacle at the cursor position.
  - **R Key:** Remove all obstacles from the canvas.

- **UI:**
  - **H Key:** Toggle the on-screen UI display.

## Code Structure

- **Raindrop Class:**  
  Manages individual raindrop particles (position, velocity, display). Raindrops are drawn as ellipses aligned with their motion.

- **Splash Class:**  
  Handles splash particles generated when raindrops impact the ground or obstacles. Splashes have multiple generations and interact to produce new splash particles.

- **Obstacle Class:**  
  Defines rectangular obstacles that can be added via mouse clicks. Obstacles influence where splashes are generated.

- **Drip Class:**  
  Simulates water drips that fall from the bottom edge of obstacles.

- **Main Functions:**
  - **setup():** Initializes canvas and particle arrays.
  - **draw():** Updates and displays all particles, handles obstacle collisions, splash interactions, and UI.
  - **handleSplashInteractions():** Processes interactions between splash particles to generate additional generations.

## Customization

You can tweak the following adjustable parameters at the top of the code:
- **Gravity, Wind Force, Rain Speed, and Rain Size** for the raindrops.
- **Splash Intensity, Splash Interaction Radius, and Splash Reproduction Rate** for splash behaviors.
- **Rain Density** to control how many raindrops are generated each frame.
- **UI Visibility and Obstacle Collision** toggles.

## Credits

This simulation was developed using Processing and demonstrates application of Computational Intelligence using particle systems and interactive elements.

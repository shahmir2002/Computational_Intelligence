# Evolutionary Algorithm for Solving the Traveling Salesman Problem (TSP)

## Overview

This project implements an evolutionary algorithm to solve the Traveling Salesman Problem (TSP). The algorithm employs various parent and survival selection strategies to evolve a population of solutions over multiple generations.

## Features

- Reads TSP instances from `.tsplib` files
- Computes Euclidean distances between cities
- Implements multiple selection strategies:
  - **Parent Selection:** Fitness Proportionate Selection (FPS), Rank-Based Selection (RBS), Binary Tournament, Truncation, and Random Selection
  - **Survival Selection:** FPS, RBS, Binary Tournament, Truncation, and Random Selection
- Uses order-based crossover and swap mutation
- Tracks and visualizes the best fitness and average fitness over generations

## Requirements

- Python 3.x
- Required libraries:
  - `matplotlib`
  - `numpy`
  - `random`
  - `math`

Install dependencies using:

```sh
pip install matplotlib numpy
```

## Usage

### Running the Algorithm

1. Place the TSP dataset file (e.g., `qa194.tsp`) in the same directory as the script.
2. Run the script with the default settings:
   ```sh
   python tsp_evolutionary.py
   ```
3. The script will iterate through generations, optimizing the TSP tour and displaying the final fitness results.

### Configurable Parameters

Modify the following parameters in the script:

```python
pop_size = 194  # Population size
num_offspring = 194  # Number of offspring per generation
generations = 400  # Number of generations
mutation_rate = 0.18  # Mutation probability per gene
iterations = 10  # Number of runs for averaging results
parent_selection = "fps"  # Options: "fps", "rbs", "binary_tournament", "truncation", "random"
survival_selection = "rbs"  # Options: "fps", "rbs", "binary_tournament", "truncation", "random"
```

### Output & Visualization

- The script prints the best fitness and best tour found.
- After completing the iterations, it plots the average and best-so-far fitness history across generations.

## Algorithm Breakdown

1. **Read TSP File**: Parses `.tsplib` files to extract city coordinates.
2. **Initialize Population**: Generates a random population of TSP tours.
3. **Evaluate Fitness**: Computes the total Euclidean distance for each tour.
4. **Parent Selection**: Selects parents based on a chosen strategy.
5. **Crossover & Mutation**: Applies order-based crossover and swap mutation.
6. **Survival Selection**: Selects the next generation based on a chosen strategy.
7. **Repeat for Generations**: Evolves the population over multiple generations.
8. **Plot Results**: Displays the convergence trend of the evolutionary algorithm.

## Example TSP Dataset Format (`qa194.tsp`)

```
NAME: qa194
TYPE: TSP
DIMENSION: 194
NODE_COORD_SECTION
1 1088 3763
2 1465 3412
3 2563 2438
...
194 2000 1500
EOF
```



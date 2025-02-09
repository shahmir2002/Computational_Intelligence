# Job Shop Scheduling Problem (JSSP) Evolutionary Algorithm

## Overview

This project implements an Evolutionary Algorithm (EA) to solve the Job Shop Scheduling Problem (JSSP). The algorithm optimizes job schedules to minimize the makespan (total time required to complete all jobs) by evolving populations of schedules over multiple generations.

## Features

- **Flexible JSSP Instance Parsing:** Reads job and machine data from a text file.
- **Genetic Operators:** Includes selection, crossover, and mutation mechanisms.
- **Multiple Selection Methods:** Supports FPS, RBS, Tournament, Truncation, and Random selection.
- **Graphical Analysis:** Plots the evolution of the makespan over generations.
- **Configurable Parameters:** Allows customization of population size, mutation rate, number of generations, and selection strategies.

## Installation

Ensure you have Python installed along with the required dependencies:

```sh
pip install numpy matplotlib
```

## Usage

1. **Prepare a JSSP instance file** with the following format:
   - The first line contains two integers: the number of jobs and machines.
   - Each subsequent line represents a job with alternating machine IDs and processing times.
2. **Run the algorithm:**

```sh
python jssp_solver.py
```

## Code Structure

- **`Operation`**\*\* Class:\*\* Represents an individual operation in a job.
- **`Job`**\*\* Class:\*\* Represents a job consisting of multiple operations.
- **`JSSPInstance`**\*\* Class:\*\* Parses and stores the JSSP data.
- **Evolutionary Algorithm Components:**
  - **Initialization:** Generates random schedules.
  - **Selection:** Different methods to select parents and survivors.
  - **Crossover (PPX):** Ensures precedence preservation in offspring.
  - **Mutation:** Random swaps to introduce diversity.
  - **Evaluation:** Calculates makespan for schedules.
- **Visualization:** Plots the average and best-so-far makespan trends over generations.

## Example Execution

Modify the `file_path` variable to point to your JSSP instance file and configure parameters as needed:

```python
file_path = "abz7.txt"
pop_size = 30
generations = 200
mutation_rate = 0.18
parent_selection = "rbs"
survivor_selection = "tournament"
```

Run the script multiple times to observe algorithm performance over iterations.

## Example .txt format for JSSP instance:



10 5
1 21 0 53 4 95 3 55 2 34
0 21 3 52 4 16 2 26 1 71
......

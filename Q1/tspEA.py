import random
import math
import matplotlib.pyplot as plt
import numpy as np

#Reading .tsplib file
def read_tsp_file(file_path):
    #reads all lines from the file into a list of strings (lines)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    #Determine where the cities data begins, store it in node_coord_start
    node_coord_start = None
    for i, line in enumerate(lines):
        if line.startswith("NODE_COORD_SECTION"):
            node_coord_start = i + 1 #mark where coordinate data begins.
            break
    
    #Making a list of tuples containing x,y coordinates of each city
    cities = []
    for line in lines[node_coord_start:-1]:  # Skip the last line (EOF)
        parts = line.strip().split()
        x,y = float(parts[1]),float(parts[2])
        cities.append((x, y))
    return cities

# Calculate Euclidean distance between two cities
def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

#Total distance in the entire tour
def total_distance(tour,cities):
    total = 0
    #add each consecutive distance according to the order in tour
    for i in range(len(tour) - 1):
        total += distance(cities[tour[i]], cities[tour[i+1]])
    #add the distance from last node to the first one to finish the tour
    total += distance(cities[tour[-1]], cities[tour[0]])

    # print("CHECK")
    # print(total)
    return(sum(distance(cities[tour[i]], cities[tour[i+1]]) for i in range(len(tour) - 1)) + distance(cities[tour[-1]], cities[tour[0]]))

#returns a random population(a list of lists)
def initialize_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]

#FLAG TRUE MEANS PARENT SELECTION
def fps_selection(population, distances, pop_size):
    fitness =  [1 / f for f in distances] #fitness will be the inverse of distance(since lower distance = higher fitness)
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness] #weights for each tour
    #flag true -> Parent selection, flag false -> Survivor selection
    selected_indices = np.random.choice(len(population), size=pop_size, p=probabilities)
    return [population[i] for i in selected_indices]

# RBS Selection
import random
def rbs_selection(population, distances, pop_size):
    # Combine population and distances into a list of tuples for sorting
    combined = list(zip(population, distances))
    
    # Sort the combined list based on distances (ascending order)
    combined.sort(key=lambda x: x[1])
    
    # Extract the sorted population and distances
    sorted_population = [x[0] for x in combined]
    sorted_distances = [x[1] for x in combined]
    
    # Assign ranks (lower distance gets higher rank)
    ranks = list(range(len(sorted_population), 0, -1))
    
    # Calculate selection probabilities based on ranks
    total_rank = sum(ranks)
    probabilities = [rank / total_rank for rank in ranks]
    
    # Survival selection: Select pop_size individuals
    selected_indices = random.choices(range(len(sorted_population)), weights=probabilities, k=pop_size)
    selected_population = [sorted_population[i] for i in selected_indices]
    return selected_population


# Binary Tournament Selection
def binary_tournament_selection(population, fitness, pop_size):
    selected = []
    for _ in range(pop_size):
        candidates = random.sample(list(zip(population, fitness)), 2) #pick two tuples
        selected.append(min(candidates, key=lambda x: x[1])[0]) #compare from the two and pick the one with lower fitness
    return selected

# Truncation Selection
def truncation_selection(population, fitness ,truncation_size=0.5):
    sorted_population = [x for _, x in sorted(zip(fitness, population))]
    return sorted_population[:int(len(population) * truncation_size)]

# Random Selection
def random_selection(population,pop_size):
    return random.sample(population, pop_size)

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    remaining = [item for item in parent2 if item not in child[start:end]]
    child[:start] = remaining[:start]
    child[end:] = remaining[start:]
    return child

# Swap mutation
def mutate(tour, mutation_rate):
    for i in range(len(tour)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(tour) - 1)
            tour[i], tour[j] = tour[j], tour[i]
    return tour

def evolutionary_algorithm(cities, pop_size, num_offspring, generations, mutation_rate, parent_selection, survival_selection):
    num_cities = len(cities)
    population = initialize_population(pop_size, num_cities)
    # print("Population -->" , population)
    best_fitness = float('inf')
    best_tour = None
    avg_fitness_history = []
    bsf_history = []

    for gen in range(generations):
        print("Gen: ", gen)
        distances = []  #create a list distances which contains total distances of each tour in the population
        for tour in population:
            distances.append(total_distance(tour, cities))
        # print("Distances --> ",  distances)
        current_best = min(distances)
        if current_best < best_fitness:
            best_fitness = current_best
            best_tour = population[distances.index(current_best)]
        # print("Best fitness", best_fitness)
        # print("Best tour", best_tour)
        avg_fitness = sum(distances) / len(distances)
        avg_fitness_history.append(avg_fitness)
        bsf_history.append(best_fitness)

        parents=[]
        # Parent selection
        if parent_selection == "fps":
            parents = fps_selection(population, distances, pop_size)
        elif parent_selection == "rbs":
            parents = rbs_selection(population, distances, pop_size)
        elif parent_selection == "binary_tournament":
            parents = binary_tournament_selection(population, distances, pop_size)
        elif parent_selection == "truncation":
            parents = truncation_selection(population, distances,truncation_size=0.5)
        elif parent_selection == "random":
            parents = random_selection(population, pop_size)
        else:
            raise ValueError("Invalid parent selection method")
        
        # print("Parents -->" ,  parents)
        # print("Parents Length" , len(parents))
        # # Create offspring
        offsprings = []
        for i in range(0, num_offspring-1, 2):
            # print("i = ", i)
            parent1, parent2 = parents[i],parents[i+1]
            # print("Parent 1 ->" , parent1,"Parent 2 ->" , parent2)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            offsprings.append(mutate(child1, mutation_rate))
            offsprings.append(mutate(child2, mutation_rate))
        
        # print("Offspring --> " , offsprings)

        # Survival selection
        offspring_distances = [total_distance(tour, cities) for tour in offsprings]
        combined_population = population + offsprings
        combined_distances = distances + offspring_distances

        if survival_selection == "fps":
            population = fps_selection(combined_population, combined_distances, pop_size)
        elif survival_selection == "rbs":
            population = rbs_selection(combined_population, combined_distances, pop_size)
        elif survival_selection == "binary_tournament":
            population = binary_tournament_selection(combined_population, combined_distances, pop_size)
        elif survival_selection == "truncation":
            population = truncation_selection(combined_population, combined_distances)
        elif survival_selection == "random":
            population = random_selection(combined_population ,pop_size)
        else:
            raise ValueError("Invalid survival selection method")
        

    return best_tour, best_fitness, avg_fitness_history, bsf_history

def plot_results(avg_fitness_history, bsf_history, title):
    plt.figure(figsize=(10, 6))
    plt.plot(avg_fitness_history, label="Average Fitness")
    plt.plot(bsf_history, label="Best-So-Far Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Tour Length)")
    plt.title(title)
    plt.legend()
    plt.show()

import numpy as np

if __name__ == "__main__":
    # Load the TSP dataset
    file_path = "qa194.tsp" 
    cities = read_tsp_file(file_path)
    print(f"Number of cities: {len(cities)}")

    # Configurable parameters
    pop_size = 194
    num_offspring = 194
    generations = 400
    mutation_rate = 0.18
    iterations = 10
    parent_selection = "fps"
    survival_selection = "rbs"

    # Storage for results
    all_avg_fitness_history = []
    all_bsf_history = []

    for i in range(iterations):
        print(f"Running iteration {i+1}/{iterations}...")
        best_tour, best_fitness, avg_fitness_history, bsf_history = evolutionary_algorithm(
            cities, pop_size, num_offspring, generations, mutation_rate, parent_selection, survival_selection
        )
        
        all_avg_fitness_history.append(avg_fitness_history)
        all_bsf_history.append(bsf_history)

    # Convert to NumPy arrays for easier averaging
    all_avg_fitness_history = np.array(all_avg_fitness_history)
    all_bsf_history = np.array(all_bsf_history)

    # Compute mean across iterations
    mean_avg_fitness = np.mean(all_avg_fitness_history, axis=0)
    mean_bsf = np.mean(all_bsf_history, axis=0)

    # Plot results
    plot_results(mean_avg_fitness, mean_bsf, f"Avg Over {iterations} Runs (Parent: {parent_selection}, Survival: {survival_selection})")

    

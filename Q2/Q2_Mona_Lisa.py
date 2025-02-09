import numpy as np
import random
import cv2
import os
import time
import concurrent.futures
from skimage.metrics import structural_similarity as ssim  # Faster fitness function

# 🔹 Constants
POPULATION_SIZE = 50  
NUM_POLYGONS = 50    
CANVAS_WIDTH = 400    
CANVAS_HEIGHT = 500
SAVE_INTERVAL = 500   
TOURNAMENT_SIZE = 10
MUTATION_RATE = 0.3

# 🔹 Load and preprocess the target image
TARGET_IMG = cv2.imread("Mona_Lisa.jpg", cv2.IMREAD_COLOR)
if TARGET_IMG is None:
    raise FileNotFoundError("Mona_Lisa.jpg not found.")
TARGET_IMG = cv2.resize(TARGET_IMG, (CANVAS_WIDTH, CANVAS_HEIGHT))
TARGET_GRAY = cv2.cvtColor(TARGET_IMG, cv2.COLOR_BGR2GRAY)

# 🔹 Create folder for saving images
if not os.path.exists("saved_images"):
    os.makedirs("saved_images")

# 🔹 Generate a random polygon
def generate_random_polygon():
    num_sides = random.randint(3, 8)
    color = tuple(random.randint(0, 255) for _ in range(3))
    alpha = random.uniform(0.2, 0.8)
    points = np.array([(random.randint(0, CANVAS_WIDTH), random.randint(0, CANVAS_HEIGHT)) for _ in range(num_sides)], dtype=np.int32)
    return color, alpha, points

# 🔹 Generate a random individual
def generate_individual():
    return [generate_random_polygon() for _ in range(NUM_POLYGONS)]

# 🔹 Optimized drawing function
def draw_individual(individual):
    # start_time = time.time()
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 255, dtype=np.uint8)

    overlay = np.zeros_like(canvas)  # Use a single overlay to avoid redundant blending
    mask = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.uint8)

    for color, alpha, points in individual:
        poly_mask = np.zeros_like(mask)
        cv2.fillPoly(poly_mask, [points], 255)
        
        overlay[poly_mask == 255] = color
        mask[poly_mask == 255] = 255  # Store blended area

    # Apply blending only once
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

    # end_time = time.time()
    # print(f"Draw Time: {end_time - start_time:.4f} sec")
    return canvas

# 🔹 Optimized fitness function using SSIM
def compute_fitness(individual):
    # start_time = time.time()
    candidate_img = draw_individual(individual)

    # Compute SSIM (better and faster than MSE)
    candidate_gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
    fitness = ssim(TARGET_GRAY, candidate_gray)

    # end_time = time.time()
    # print(f"Fitness Computation Time: {end_time - start_time:.4f} sec")
    return fitness

# 🔹 Parallel fitness computation
def compute_fitness_parallel(population):
    # start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        fitness_values = list(executor.map(compute_fitness, population))
    # end_time = time.time()
    # print(f"Population Fitness Time: {end_time - start_time:.4f} sec")
    return fitness_values

# 🔹 **Tournament Selection**
def tournament_selection(population, fitness_values):
    return max(random.sample(list(zip(population, fitness_values)), TOURNAMENT_SIZE), key=lambda x: x[1])[0]

# 🔹 **Top 20 Always Survive**
def select_survivors(population, fitness_values):
    ranked = sorted(zip(population, fitness_values), key=lambda x: x[1], reverse=True)
    return [ind for ind, _ in ranked[:20]]  # Keep only the best 20 individuals

# 🔹 **Optimized Crossover**
def crossover(parent1, parent2):
    point = random.randint(int(0.2 * NUM_POLYGONS), int(0.8 * NUM_POLYGONS))
    return parent1[:point] + parent2[point:]

# 🔹 **Optimized Mutation**
def mutate(individual):
    return [generate_random_polygon() if random.random() < MUTATION_RATE else polygon for polygon in individual]

# 🔹 **Save Best Individual Image**
def save_best_image(best_individual, generation):
    best_img = draw_individual(best_individual)
    filename = f"saved_images_timeOPT/gen_{generation}.png"
    cv2.imwrite(filename, best_img)

# 🔹 **Main Evolutionary Loop**
def run_evolution(max_generations=10000, target_fitness=0.9):
    population = [generate_individual() for _ in range(POPULATION_SIZE)]
    best_fitness = -1.0

    for gen in range(max_generations):
        # gen_start_time = time.time()

        fitness_values = compute_fitness_parallel(population)
        avg_fitness, gen_best = np.mean(fitness_values), max(fitness_values)
        best_fitness = max(best_fitness, gen_best)

        # Print every 50 generations
        if gen % 50 == 0:
            print(f"Generation {gen}: Best Fitness: {best_fitness:.4f}, Avg Fitness: {avg_fitness:.4f}")

        # Save an image every 500 generations
        if gen % 50 == 0:
            best_individual = population[np.argmax(fitness_values)]
            save_best_image(best_individual, gen)

        if best_fitness >= target_fitness:
            print("Convergence reached!")
            print(f"Generation {gen}: Best Fitness: {best_fitness:.4f}, Avg Fitness: {avg_fitness:.4f}")
            save_best_image(best_individual, gen)
            break

        survivors = select_survivors(population, fitness_values)  # Keep top 20
        new_population = survivors[:]

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)
            child = mutate(crossover(parent1, parent2))
            new_population.append(child)

        population = new_population
        # gen_end_time = time.time()
        # print(f"Total Generation Time: {gen_end_time - gen_start_time:.4f} sec")

if __name__ == "__main__":
    run_evolution(max_generations=10000, target_fitness=0.99)

import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Class to represent an operation
class Operation:
    def __init__(self, job_id: int, operation_id: int, machine_id: int, processing_time: int):
        self.job_id = job_id
        self.operation_id = operation_id
        self.machine_id = machine_id
        self.processing_time = processing_time

# Class to represent a job
class Job:
    def __init__(self, job_id: int, operations: List[Operation]):
        self.job_id = job_id
        self.operations = operations

# Class to represent a JSSP instance
class JSSPInstance:
    def __init__(self, num_jobs: int, num_machines: int, jobs: List[Job]):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.jobs = jobs

    @staticmethod
    def from_file(file_path: str):
        with open(file_path, 'r') as file:
            lines = file.readlines()
    
        # Find the line with the number of jobs and machines
        num_jobs, num_machines = None, None
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.replace(" ", "").isdigit():  # Check if the line contains only numbers
                parts = stripped_line.split()
                if len(parts) == 2:  # Ensure there are exactly two numbers
                    num_jobs, num_machines = map(int, parts)
                    break

        if num_jobs is None or num_machines is None:
            raise ValueError("Could not find the number of jobs and machines in the file.")

        # print(f"Number of jobs: {num_jobs}, Number of machines: {num_machines}")

        # Parse jobs and operations
        jobs = []
        for job_id in range(num_jobs):
            operation_data = list(map(int, lines[1 + job_id].strip().split()))
            # print(f"Job {job_id} data: {operation_data}")  # Debug: Print operation data

            # Validate the length of operation_data
            if len(operation_data) != 2 * num_machines:
                raise ValueError(
                    f"Invalid operation data for job {job_id}. Expected {2 * num_machines} values, got {len(operation_data)}."
                )
            
            operations = []
            for op_id in range(num_machines):
                machine_id = operation_data[2 * op_id]
                processing_time = operation_data[2 * op_id + 1]
                operations.append(Operation(job_id, op_id, machine_id, processing_time))
            jobs.append(Job(job_id, operations))

        return JSSPInstance(num_jobs, num_machines, jobs)

# Function to calculate the makespan of a schedule
def calculate_makespan(schedule: List[int], instance: JSSPInstance) -> int:
    # Initialize machine and job completion times
    machine_times = [0] * instance.num_machines
    job_times = [0] * instance.num_jobs

    for operation_id in schedule:
        job = instance.jobs[operation_id // instance.num_machines]
        operation = job.operations[operation_id % instance.num_machines]

        # Update machine and job completion times
        start_time = max(machine_times[operation.machine_id], job_times[job.job_id])
        end_time = start_time + operation.processing_time
        machine_times[operation.machine_id] = end_time
        job_times[job.job_id] = end_time

    return max(machine_times)

# Function to initialize a random population
def initialize_population(pop_size: int, instance: JSSPInstance) -> List[List[int]]:
    population = []
    for _ in range(pop_size):
        # Generate a random schedule respecting operation order
        schedule = []
        for job in instance.jobs:
            schedule.extend([job.job_id * instance.num_machines + op.operation_id for op in job.operations])
        random.shuffle(schedule)
        population.append(schedule)
    return population

# Precedence Preservative Crossover (PPX)
def ppx_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    child = []
    i, j = 0, 0
    while i < len(parent1) and j < len(parent2):
        if random.random() < 0.5:
            if parent1[i] not in child:
                child.append(parent1[i])
            i += 1
        else:
            if parent2[j] not in child:
                child.append(parent2[j])
            j += 1
    # Add remaining operations
    for op in parent1[i:] + parent2[j:]:
        if op not in child:
            child.append(op)
    return child

# Swap mutation
def swap_mutation(schedule: List[int], mutation_rate: float) -> List[int]:
    for i in range(len(schedule)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(schedule) - 1)
            schedule[i], schedule[j] = schedule[j], schedule[i]
    return schedule

def fps_selection(population: List[List[int]], fitness: List[float], num_selected: int) -> List[List[int]]:

    # Convert fitness to selection probabilities
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]

    # Select individuals based on probabilities
    selected_indices = np.random.choice(len(population), size=num_selected, p=probabilities)
    return [population[i] for i in selected_indices]

def rbs_selection(population: List[List[int]], fitness: List[float], num_selected: int) -> List[List[int]]:

    # Rank the population based on fitness (lower fitness = higher rank)
    ranked_population = [x for _, x in sorted(zip(fitness, population), key=lambda pair: pair[0])]

    # Assign ranks (1 is the best, N is the worst)
    ranks = list(range(len(population), 0, -1))

    # Calculate selection probabilities based on ranks
    total_rank = sum(ranks)
    probabilities = [rank / total_rank for rank in ranks]

    # Select individuals based on probabilities
    selected_indices = np.random.choice(len(population), size=num_selected, p=probabilities)
    return [ranked_population[i] for i in selected_indices]

def binary_tournament_selection(population,makespans, num_selected):
    parents = []
    for _ in range(num_selected):
        candidates = random.sample(list(zip(population, makespans)), 2)
        parents.append(min(candidates, key=lambda x: x[1])[0])
    return parents

def truncation_selection(population, makespans, num_selected):
    # Sort population by fitness (lower makespan is better)
    sorted_population = [x for _, x in sorted(zip(makespans, population))]
    # Select the top 'num_selected' individuals
    return sorted_population[:num_selected]

def random_selection(population,pop_size):
    return random.sample(population, pop_size)

# Evolutionary Algorithm for JSSP
def evolutionary_algorithm(
    instance: JSSPInstance,
    pop_size: int,
    num_offspring: int,
    generations: int,
    mutation_rate: float,
    parent_selection: str,
    survivor_selection: str,
):
    population = initialize_population(pop_size, instance)
    best_makespan = float('inf')
    best_schedule = None
    avg_makespan_history = []
    bsf_history = []

    for gen in range(generations):
        # Evaluate fitness
        makespans = [calculate_makespan(schedule, instance) for schedule in population]
        current_best = min(makespans)
        if current_best < best_makespan:
            best_makespan = current_best
            best_schedule = population[makespans.index(current_best)]
        avg_makespan = sum(makespans) / len(makespans)
        avg_makespan_history.append(avg_makespan)
        bsf_history.append(best_makespan)

        # Parent selection
        if parent_selection == "fps":
            parents = fps_selection(population, makespans, num_offspring)
        elif parent_selection == "rbs":
            parents = rbs_selection(population, makespans, num_offspring)
        elif parent_selection == "tournament":
            parents = binary_tournament_selection(population, makespans, num_offspring)
        elif parent_selection == "truncation":
            parents = truncation_selection(population, makespans, num_offspring)
        elif parent_selection == "random":
            parents = random_selection(population,num_offspring)
        else:
            raise ValueError("Invalid parent selection method")
        
        
        # Create offspring
        offspring = []
        for i in range(0, num_offspring - 1, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1 = ppx_crossover(parent1, parent2)
            child2 = ppx_crossover(parent2, parent1)
            offspring.append(swap_mutation(child1, mutation_rate))
            offspring.append(swap_mutation(child2, mutation_rate))

        # Survival selection
        offspring_makespans = [calculate_makespan(schedule, instance) for schedule in offspring]
        combined_population = population + offspring
        combined_makespans = makespans + offspring_makespans

        if survivor_selection == "fps":
            population = fps_selection(combined_population, combined_makespans, pop_size)
        elif survivor_selection == "rbs":
            population = rbs_selection(combined_population, combined_makespans, pop_size)
        elif survivor_selection == "tournament":
            population = binary_tournament_selection(combined_population, combined_makespans, pop_size)
        elif survivor_selection == "truncation":
            population = truncation_selection(combined_population, combined_makespans, pop_size)
        elif survivor_selection == "random":
            population = random_selection(combined_population,pop_size)
        else:
            raise ValueError("Invalid survivor selection method")

    return best_schedule, best_makespan, avg_makespan_history, bsf_history
# Function to plot results
def plot_results(avg_makespan_history, bsf_history, title):
    plt.figure(figsize=(10, 6))
    minavg = min(avg_makespan_history)
    minbsf= min(bsf_history)
    plt.plot(avg_makespan_history, label=f"Average Makespan (Min = {minavg} )")
    plt.plot(bsf_history, label=f"Best-So-Far Makespan (Min = {minbsf} ) ")
    plt.xlabel("Generation")
    plt.ylabel("Makespan")
    plt.title(title)
    plt.legend()
    plt.show()

# Main function
import numpy as np

if __name__ == "__main__":
    # Load JSSP instance
    file_path = "abz7.txt"  # Replace with the path to your JSSP instance file
    instance = JSSPInstance.from_file(file_path)

    # Configurable parameters
    pop_size = 30
    num_offspring = 30
    generations = 200
    mutation_rate = 0.18
    parent_selection = "rbs"
    survivor_selection = "tournament"
    iterations = 10  # Number of runs

    # Storage for results
    all_avg_makespan_history = []
    all_bsf_history = []

    for i in range(iterations):
        print(f"Running iteration {i+1}/{iterations}...")
        best_schedule, best_makespan, avg_makespan_history, bsf_history = evolutionary_algorithm(
            instance, pop_size, num_offspring, generations, mutation_rate, parent_selection, survivor_selection
        )

        all_avg_makespan_history.append(avg_makespan_history)
        all_bsf_history.append(bsf_history)

    # Convert to NumPy arrays for easier averaging
    all_avg_makespan_history = np.array(all_avg_makespan_history)
    all_bsf_history = np.array(all_bsf_history)

    # Compute mean across iterations
    mean_avg_makespan = np.mean(all_avg_makespan_history, axis=0)
    mean_bsf = np.mean(all_bsf_history, axis=0)

    # Plot results
    plot_results(mean_avg_makespan, mean_bsf, f"Avg Over {iterations} Runs (Parent: {parent_selection}, Survival: {survivor_selection})")

    

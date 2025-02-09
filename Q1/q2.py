import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple

# Define Job-Shop Scheduling Problem (JSSP) class
class JobShopProblem:
    def _init_(self, num_jobs, num_machines, processing_times):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.processing_times = processing_times  # Matrix of size [jobs x machines]

# Define Chromosome Representation
class Schedule:
    def _init_(self, problem, sequence=None):
        self.problem = problem
        self.sequence = sequence if sequence else self.random_sequence()
        self.fitness = self.compute_fitness()
    
    def random_sequence(self):
        sequence = []
        for j in range(self.problem.num_jobs):
            sequence.extend([j] * self.problem.num_machines)
        random.shuffle(sequence)
        return sequence
    
    def compute_fitness(self):
        # Compute Makespan (total time to complete all jobs)
        completion_times = np.zeros((self.problem.num_jobs, self.problem.num_machines))
        machine_available_time = np.zeros(self.problem.num_machines)
        job_available_time = np.zeros(self.problem.num_jobs)

        for job in self.sequence:
            machine = self.problem.processing_times[job].index(min(self.problem.processing_times[job]))
            start_time = max(machine_available_time[machine], job_available_time[job])
            completion_times[job][machine] = start_time + self.problem.processing_times[job][machine]
            machine_available_time[machine] = completion_times[job][machine]
            job_available_time[job] = completion_times[job][machine]
        
        return np.max(completion_times)

# Genetic Algorithm Implementation
class EvolutionaryAlgorithm:
    def _init_(self, problem, population_size=30, generations=50, mutation_rate=0.5, iterations=10):
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.iterations = iterations
        self.population = [Schedule(problem) for _ in range(population_size)]

    def select_parents(self, method='tournament'):
        if method == 'tournament':
            return max(random.sample(self.population, 2), key=lambda x: x.fitness)
        elif method == 'roulette':
            total_fitness = sum(ind.fitness for ind in self.population)
            pick = random.uniform(0, total_fitness)
            current = 0
            for ind in self.population:
                current += ind.fitness
                if current > pick:
                    return ind
        return random.choice(self.population)
    
    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1.sequence) - 1)
        child_sequence = parent1.sequence[:point] + parent2.sequence[point:]
        return Schedule(self.problem, sequence=child_sequence)
    
    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual.sequence)), 2)
            individual.sequence[idx1], individual.sequence[idx2] = individual.sequence[idx2], individual.sequence[idx1]
        return individual
    
    def evolve(self):
        best_fitness_per_gen = []
        avg_fitness_per_gen = []

        for _ in range(self.generations):
            offspring = [self.mutate(self.crossover(self.select_parents(), self.select_parents())) for _ in range(self.population_size // 2)]
            self.population.extend(offspring)
            self.population.sort(key=lambda x: x.fitness)
            self.population = self.population[:self.population_size]
            best_fitness_per_gen.append(self.population[0].fitness)
            avg_fitness_per_gen.append(sum(ind.fitness for ind in self.population) / self.population_size)
        
        return best_fitness_per_gen, avg_fitness_per_gen

# Test with a sample JSSP instance
if _name_ == "_main_":
    processing_times = [
        [3, 2, 2],
        [2, 1, 4],
        [4, 3, 3]
    ]  # Example with 3 jobs and 3 machines

    jssp = JobShopProblem(num_jobs=3, num_machines=3, processing_times=processing_times)
    ea = EvolutionaryAlgorithm(jssp)
    best_fitness, avg_fitness = ea.evolve()
    
    # Create and display results table
    results_df = pd.DataFrame({
        'Generation': list(range(1, len(best_fitness) + 1)),
        'Best Fitness': best_fitness,
        'Average Fitness': avg_fitness
    })
    print(results_df)
    
    # Plot results
    plt.figure(figsize=(10,5))
    plt.plot(results_df['Generation'], results_df['Best Fitness'], label='Best Fitness', marker='o')
    plt.plot(results_df['Generation'], results_df['Average Fitness'], label='Average Fitness', linestyle='dashed', marker='s')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution Over Generations')
    plt.legend()
    plt.show()
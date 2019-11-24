#References:
# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
# https://www.geeksforgeeks.org/confusion-matrix-machine-learning/
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
from load_data import read_data
from load_data import clean_data
from feature_selection import fitness_function
from feature_selection import generate_random_solution
import pandas as pd
import random
import time

# Set random seed
random.seed(1)
time.sleep(1)

dataset = read_data()
dataset = clean_data(dataset)

print("Fitness if using all features", fitness_function(dataset))

# Use the standard 13 features as a benchmark to measure against
standard_dataset = dataset[["age", "sex", "CP", "trestbps", "chol", "FBS", "restecg",
"thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]]

print("Fitness of the standard features typically included", fitness_function(standard_dataset))

population = []
population_fitness = []
# Start off by generating 10 random solutions
for solution_idx in range(10):
    solution = generate_random_solution(dataset)
    population.append(solution)
    fitness = fitness_function(solution)
    population_fitness.append(fitness_function(solution))

max_fitness = (max(population_fitness))
max_index = population_fitness.index(max_fitness)
best_sample = population[max_index]

print(max_fitness)
print(best_sample)

#References:
# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
# https://www.geeksforgeeks.org/confusion-matrix-machine-learning/
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# https://stackoverflow.com/questions/13668393/python-sorting-two-lists
# Python's pandas documentation
# https://gist.github.com/rlabbe/ea3444ef48641678d733
# https://medium.com/dunder-data/selecting-subsets-of-data-in-pandas-6fcd0170be9c
# https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns

from load_data import read_data
from load_data import clean_data
from feature_selection import fitness_function
from feature_selection import generate_random_solution
from feature_selection import sort_pop
from feature_selection import random_crossover
from feature_selection import mutation
from feature_selection import half_crossover
from feature_selection import remove_duplicates
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

# Set random seed
random.seed(1)
time.sleep(1)

dataset = read_data()
# dataset = clean_data(dataset)
print("!!!!!!!!!")
print(len(dataset.index))

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

population_sorted, population_fitness_sorted = sort_pop(population, population_fitness)

# For graphing
fig = plt.gcf()
fig.show()
fig.canvas.draw()
iteration = 0
best_fitnesses = []
iterations = []

# while there are still missclassifications that exist
while (population_fitness_sorted[0] < 1):
    # Always keep the best solution
    population[0] = population_sorted[0]

    # TODO: figure out why the fitness sometimes decreases if we
    # are always taking the most fit individual
    # Random crossover of best two solutions
    population[1] = remove_duplicates(random_crossover(population[0], population[1]), dataset)

    # Crossover of two random solutions
    solution1 = random.randrange(len(population))
    solution2 = random.randrange(len(population))
    while (solution2 == solution1):
        solution2 = random.randrange(len(population))
    population[2] = remove_duplicates(random_crossover(population_sorted[solution1], population_sorted[solution2]), dataset)

    # Three new random solutions
    population[3] = generate_random_solution(dataset)
    population[4] = generate_random_solution(dataset)
    population[5] = generate_random_solution(dataset)

    # Mutation to most fit solution
    population[6] = remove_duplicates(mutation(population[0], dataset), dataset)
    population[7] = remove_duplicates(mutation(population[0], dataset), dataset)

    # Mutation to second most fit solution
    population[8] = remove_duplicates(mutation(population[1], dataset), dataset)

    # Half crossover of two best solutions
    population[9] = remove_duplicates(half_crossover(population[0], population[1]), dataset)

    # Sort the individuals in the population
    population_fitness = []

    for index, individual in enumerate(population):
        if (index == 0):
            new_fitness = fitness_function(individual)
            if (new_fitness > population_fitness_sorted[0]):
                population_fitness.append(new_fitness)
            else:
                population_fitness.append(population_fitness_sorted[0])
        else:
            population_fitness.append(fitness_function(individual))
    population_sorted, population_fitness_sorted = sort_pop(population, population_fitness)

    if population_fitness_sorted[0] == 1:
        print("No missclassifications!")
        print("Feature set:")
        print(population_sorted[0].columns)
    iterations.append(iteration)
    best_fitnesses.append(max(population_fitness))

    plt.title("Fitness of Solution")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.plot(iterations, best_fitnesses)

    fig.canvas.draw()
    plt.savefig('features.png')

    iteration += 1

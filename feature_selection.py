from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import random
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import copy


# Set random seed
random.seed(1)
time.sleep(1)

# Calculate the fitness of using the selected features
# The fitness is defined as the classification accuracy
# when using k nearest neighbors as the classification model
def fitness_function(dataset):
    y = dataset["num"].values
    dataset = dataset.drop(columns = ["num"])
    X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.60)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    classifier =  KNeighborsClassifier(3)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return(accuracy_score(y_test, y_pred))


## Function to generate a completely random solution ##
def generate_random_solution (dataset):
    num_features = 13
    features_selected = []
    for feature_idx in range(num_features):
        feature = random.randrange(len(dataset.columns))
        #Make sure its not a feature we already selected
        while (feature in features_selected):
            feature = random.randrange(len(dataset.columns))
        features_selected.append(feature)
    selected_data = dataset.iloc[:, features_selected]
    # Always need to have classification column in dataframe
    pd.options.mode.chained_assignment = None  # default='warn'
    selected_data["num"] = dataset["num"]
    return selected_data

# Creates a crossover of two solutions, where the crossover point is chosen at random
def random_crossover(dataset1, dataset2):
    # crossover_point = random.randrange(1, len(dataset1.columns))
    # y = dataset1["num"].values
    # dataset1 = dataset1.drop(columns = ["num"])
    # dataset2 = dataset2.drop(columns = ["num"])
    return dataset1

# Creates a crossover of two solution, where the crossover point is always selected
# such that half the solution is from solution 1 and the other half is from solution 2
def half_crossover(dataset1, dataset2):
    return dataset1

# Creates two random mutations in the feature set
def mutation(dataset):
    return dataset

# Sort the population based on fitness
def sort_pop(population, population_fitness):
    population_fitness_sorted = copy.deepcopy(population_fitness)
    population_sorted = copy.deepcopy(population)
    population_fitness_sorted, population_sorted = [list(x) for x in zip(*sorted(zip(population_fitness_sorted, population_sorted), key=lambda pair: pair[0], reverse=True))]
    return population_sorted, population_fitness_sorted

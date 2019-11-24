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

# Set random seed
random.seed(1)
time.sleep(1)

# Calculate the fitness of using the selected features
# The fitness is defined as the classification accuracy
# when using k nearest neighbors as the classification model
def fitness_function(dataset):
    y = dataset["num"].values
    dataset.drop(columns = ["num"])

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

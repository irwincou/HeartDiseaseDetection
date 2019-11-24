import pandas as pd
import csv

def read_data():
    dataset = []
    # Every 10 lines are one row in the data table
    dataset_row = []
    with open('cleveland.data') as file:
        reader = csv.reader(file, delimiter=' ')
        for line in reader:
            #delete name field
            if ("name" in line):
                del line[4]
                line = [float(i) for i in line]
                dataset_row += line
                dataset.append(dataset_row)
                dataset_row= []
            else:
                line = [float(i) for i in line]
                dataset_row += line

    # columns of the dataset
    column_names = ["ID", "SSN", "age", "sex", "painloc", "painexer", "relrest", "pncaden",
    "CP", "trestbps", "htn", "chol", "smoke", "cigs", "years", "FBS", "DM", "famhist",
    "restecg", "ekgmo", "ekgday", "ekgyr", "dig", "prop", "nitr", "pro", "diuretic", "proto",
    "thaldur", "thaltime", "met", "thalach", "thalrest", "tpeakbps", "tpeakbpd", "dummy", "trestbpd",
    "exang", "xhypo", "oldpeak", "slope", "rldv5", "rldv5e", "ca", "restckm", "exerckm",
    "restef", "restwm", "exeref", "exerwm", "thal", "thalsev", "thalpul", "earlobe",
    "cmo", "cday", "cyr", "num", "lmt", "ladprox", "laddist", "diag", "cxmain", "ramus",
    "om1", "om2", "rcaprox", "rcadist", "lvx1", "lvx2", "lvx3", "lvx4", "lvf", "cathef",
    "junk"]


    df = pd.DataFrame(dataset, columns=column_names)

    # Now get rid of all the columns that have the description "not used" in the
    # data description
    df = df.drop(columns = ["thalsev", "thalpul", "earlobe", "lvx1", "lvx2"])
    df = df.drop(columns = ["lvx3", "lvx4", "lvf", "cathef", "junk"])

    # Get rid of ID number, SSN, dummy column
    df = df.drop(columns = ["ID", "SSN", "dummy"])

    # Get rid of columns that have the description "irrelevant" in the data description
    df = df.drop(columns = ["restckm", "exerckm"])

    return df

# For now, just delete the rows that contain a nan
def clean_data(dataset):
    dataset = dataset.dropna()
    return(dataset)

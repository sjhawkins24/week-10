import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

#Exercise 1
#import data 
data = pd.read_csv("https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv")

#Set up model inputs 
X = np.array(data["100g_USD"]).reshape(-1, 1)
y = np.array(data["rating"])

#Fit model 
model = LinearRegression()
model.fit(X,y)

#Save model in a pickle 
filename = "model_1.pickle"
with open(filename, "wb") as file:
    pickle.dump(model, file)


#Exercise 2
#Transform the categorical variable to numeric 
#This seemed more elegant than creating a new function to do the recode 
le = LabelEncoder()
data["roast_cat"] = le.fit_transform(data["roast"])

#Set up input data
X = data[["100g_USD", "roast_cat"]].to_numpy()
y = np.array(data["rating"])

#Fit model 
model = DecisionTreeClassifier()
model.fit(X,y)

#Save in pickle 
filename = "model_2.pickle"
with open(filename, "wb") as file:
    pickle.dump(model, file)
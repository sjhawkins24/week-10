import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
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
filename = "model_1.pkl"
with open(filename, "wb") as file:
    pickle.dump(model, file)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#import dataset
df = pd.read_csv("water_potability.csv")
#print(df.head(3))
columns = [i for i in df.columns]
#print(columns)

#handelling null values
df['ph'] = df['ph'].fillna(method='bfill')
df['Sulfate'] = df['Sulfate'].fillna(method='ffill')
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(method='ffill')
#print(df.isnull().sum())

#print(df.info())
#print(df.describe())

x = df.drop('Potability',axis=1)
y = df['Potability']
#print(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
tree = RandomForestClassifier(n_estimators=200)
tree.fit(x_train,y_train)

prediction = tree.predict(x_test)
#print(prediction)

from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test,prediction))
#print(columns)
ph = input("Enter ph of water = ")
Hardness = input("Enter Hardness of water = ")
Solids = input("Enter Solids conc. of water = ")
Chloramines = input("Enter Chloramines of water = ")
Sulfate = input("Enter Sulfate conc. of water = ")
Conductivity = input("Enter Conductivity of water = ")
Organic_carbon = input("Enter Organic_carbon conc. of water = ")
Trihalomethanes = input("Enter Trihalomethanes conc. of water = ")
Turbidity = input("Enter Turbidity of water = ")
def potabiity_of_water(a,b,c,d,e,f,g,h,i):
    out = tree.predict([[a,b,c,d,e,f,g,h,i]])
    return out
result = potabiity_of_water(ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity)
if result == 0:
    print("water is not potable")
else:
    print("water is potable")
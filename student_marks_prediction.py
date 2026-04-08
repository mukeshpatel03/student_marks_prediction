import numpy as np
from sklearn.linear_model import LinearRegression

#Data
hours = np.array([1,2,3,4,5]).reshape(-1,1)
marks = [30,40,50,60,70]

#Model
model = LinearRegression()
model.fit(hours, marks)

#Take input from user
study_hours = float(input("Enter number of study hours: "))

#Convert to 2D array
user_input = np.array([[study_hours]])

#Predict
prediction = model.predict(user_input)

print("Predicted Marks:", prediction[0])
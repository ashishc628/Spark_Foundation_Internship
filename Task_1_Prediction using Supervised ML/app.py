 
import pickle
import streamlit as st

# loading the trained model
import pickle
import streamlit as st
pickle_in = open('regressor.pkl', 'rb') 
regressor = pickle.load(pickle_in)
@st.cache()

#Importing All major libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
#Load the dataset
dataFrame = pd.read_csv('Dataset.csv')

# Now we have to divide Our data set into Dependent and Independent Variables.X is feature or its independent variable here its hours
# It will not depend on the result and y is dependent variable for this variable we have to done prediction on this. so score is dependent variable 

X = dataFrame.iloc[:,:-1].values
y = dataFrame.iloc[:,1].values
# X = dataFrame.iloc[:,0].values.reshape(-1,1)
# y = dataFrame.iloc[:,1].values.reshape(-1,1)

# Now we have to split the dataset into training and testing so we will use skcit learn library 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Now Feature scaling is not required in linear regression so 
# Now I have fit the training data in Linear Regression Algorithm
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)










# defining the function which will make the prediction using the data which the user inputs 
def prediction(hours):   
 
    # Pre-processing user input    
    raw_input = input('enter hours')
    k = eval(raw_input)
    hours = [[k]]
    if hours == [[k]]:
        own_pred = regressor.predict(hours)
        print("No of Hours = {}".format(hours))
        print("Predicted Score = {}".format(own_pred[0]))
        from sklearn import metrics  
        print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_predictor)) 
        print('R squared value:',metrics.explained_variance_score(y_test, y_predictor)) 
 
 """   # Making predictions 
    prediction = classifier.predict( 
        [[Gender, Married, ApplicantIncome, LoanAmount, Credit_History]])
     
    if prediction == 0:
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred"""
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Hours = st.number_input("Total loan amount")
  
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(hours) 
        st.success('Your loan is {}'.format(result))
        print(LoanAmount)
     
if __name__=='__main__': 
    main()

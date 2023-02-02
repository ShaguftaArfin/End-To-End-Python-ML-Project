# app.py
import numpy as np
import pickle
import streamlit as st
import requests
from PIL import Image
# from streamlit_option_menu import option_menu


# # Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
# st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")



# # side bar menu
# with st.sidebar:
#     selected = option_menu(
#         menu_title=None,
#         options=["home"],
#         icons= ["hosue", "book", "envelop"],
#         menu_icon= "cast",
#         default_index=0,
#         orientation="horizontal" )

# if selected == "home":
#     st.title(f'Hi, I am Shagufta :wave:')


# loading the saved model
loaded_model = pickle.load(open('/Users/hesham/Desktop/nlp_project_folder/eda_sidha/trained_model.sav', 'rb'))


# creating a function for Prediction
def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'

  
def main():
    # giving a title
    st.title('Welcome to my website check diabetise:')
    # getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('What is your Glucose Level')
    BloodPressure = st.text_input('What is your Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('What is your Insulin Level')
    BMI = st.text_input('What is your BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()

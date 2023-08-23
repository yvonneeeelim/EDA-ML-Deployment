# Import Libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from streamlit_autorefresh import st_autorefresh


# Set Page configuration
# Read more at https://docs.streamlit.io/1.6.0/library/api-reference/utilities/st.set_page_config
st.set_page_config(page_title='Predict Flower Species', page_icon='🌷', layout='wide', initial_sidebar_state='expanded')

# Set title of the app
st.title('🌷 Predict Flower Species')

# Auto-refresh every 5 seconds
count = st_autorefresh(interval=5000)
 
# Create a function to retreive latest data from the Google Sheet
def retrieveData():
    
    # Set up Google Sheets credentials
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('gsheetapi.json', scope)

    # Connect to Google Sheets
    client = gspread.authorize(credentials)

    # Open the desired sheet
    sheet = client.open_by_key('1gMvu7MJ4inAuaTIgynkm0cPr2QIaQNqjkmOtwPxZBL8').get_worksheet(0)

    # Fetch data from the sheet
    data = pd.DataFrame(sheet.get_all_values())
    
    # Set the first row of values as the headers
    data.columns = data.iloc[0]
    
    # Remove the first row of values from the data frame
    data = data[1:]
   
    return data


# Load data
df = retrieveData()
df.reset_index(inplace=True)

# Set input widgets
st.sidebar.subheader('Select flower attributes')
sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.8)
sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.1)
petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 3.8)
petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 1.2)

# Separate to X and y
X = df.drop(['index', 'Species'], axis=1)
y = df.Species

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = RandomForestClassifier(max_depth=2, max_features=4, n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Generate prediction based on user selected attributes
y_pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

# Display EDA
st.subheader('Exploratory Data Analysis')

# Display full data frame
st.dataframe(df)

# Compute mean values for each species
groupby_species_mean = df.groupby('Species', as_index=False).mean()
groupby_species_mean.columns = ['Species', 'Value']

# Display the computed mean values
st.write(groupby_species_mean)

# Create a bar chart using the computed mean values
st.bar_chart(data=groupby_species_mean, x='Species', y='Value')

# Print input features
st.subheader('Variables in Data Set')
input_feature = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
st.write(input_feature)

# Print predicted flower species
st.subheader('Prediction')
st.metric('Predicted Flower Species is :', y_pred[0], '')
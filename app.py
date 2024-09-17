import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Title and description
st.title("Journey Status Prediction App")
st.write("This app predicts the Journey Status using machine learning!")

# Load your dataset
@st.cache_data
def load_data():
    # Assuming you have a CSV or can generate the dataset dynamically
    data = pd.read_csv(r"C:\Users\Kaybee\Videos\CAPSTONE_PROJECT\railway.csv")  # Replace with actual data
    data["Reason for Delay"].fillna(value= 'None', inplace= True)
    data["Actual Arrival Time"].fillna(value= 'Unknown', inplace= True)
    data["Railcard"].fillna(value= 'none', inplace= True)
    data["Reason for Delay"] = data["Reason for Delay"].replace('Signal failure', 'Signal Failure')
    return data

data = load_data()


# 27481 - 4172
# Display dataset
st.subheader('Dataset Preview')
st.write(data.head())

# Preprocessing
cat_variable = ['Purchase Type', 'Payment Method', 'Railcard', 'Ticket Class', 
                'Ticket Type', 'Departure Station', 'Arrival Destination', 'Reason for Delay']
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(data[cat_variable])
one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(cat_variable))

scaler = StandardScaler()
one_hot_encoded_df['Price'] = scaler.fit_transform(data[['Price']])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["Journey Status"])

X = one_hot_encoded_df

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Model Training (You can adjust to suit your selected model)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Input features from the user
st.sidebar.subheader("Provide Input for Prediction")

# Taking user input for all features
purchase_type = st.sidebar.selectbox('Purchase Type', data['Purchase Type'].unique())
payment_method = st.sidebar.selectbox('Payment Method', data['Payment Method'].unique())
railcard = st.sidebar.selectbox('Railcard', data['Railcard'].unique())
ticket_class = st.sidebar.selectbox('Ticket Class', data['Ticket Class'].unique())
ticket_type = st.sidebar.selectbox('Ticket Type', data['Ticket Type'].unique())
departure_station = st.sidebar.selectbox('Departure Station', data['Departure Station'].unique())
arrival_destination = st.sidebar.selectbox('Arrival Destination', data['Arrival Destination'].unique())
reason_for_delay = st.sidebar.selectbox('Reason for Delay', data['Reason for Delay'].unique())
price = st.sidebar.number_input('Price', min_value=0.0, max_value=500.0, step=1.0)

# Create input DataFrame for prediction
input_data = pd.DataFrame({
    'Purchase Type': [purchase_type],
    'Payment Method': [payment_method],
    'Railcard': [railcard],
    'Ticket Class': [ticket_class],
    'Ticket Type': [ticket_type],
    'Departure Station': [departure_station],
    'Arrival Destination': [arrival_destination],
    'Reason for Delay': [reason_for_delay],
    'Price': [price]
})

# Apply transformations
input_data_encoded = encoder.transform(input_data[cat_variable])
input_data_scaled = scaler.transform(input_data[['Price']])
input_combined = pd.concat([pd.DataFrame(input_data_encoded), pd.DataFrame(input_data_scaled)], axis=1)

# Make predictions
if st.button('Predict Journey Status'):
    prediction = model.predict(input_combined)
    st.write(f"Predicted Journey Status: {label_encoder.inverse_transform(prediction)[0]}")

# Evaluate accuracy on test set
st.subheader("Model Evaluation")
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
st.write(f"Test Accuracy: {accuracy:.2f}")

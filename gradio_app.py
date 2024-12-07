import gradio as gr
import joblib
import pandas as pd
import os

# Load the trained model
path = os.path.join(os.path.dirname(__file__), 'flight_price_model.pkl')
model = joblib.load(path)

def preprocess_input(data):
    # Create dummy variables for categorical columns
    categorical_columns = ['airline', 'source_city', 'departure_time',
                            'stops', 'arrival_time', 'destination_city', 'class']

    # Apply one-hot encoding
    data_encoded = pd.get_dummies(data, columns=categorical_columns)

    # Ensure all columns from training are present
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in data_encoded.columns:
            data_encoded[col] = 0

    # Reorder columns to match training data
    data_encoded = data_encoded[expected_columns]

    return data_encoded

# Define the prediction function
def predict_price(airline, source_city, departure_time, stops, arrival_time, destination_city, travel_class):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'airline': [airline],
        'source_city': [source_city],
        'departure_time': [departure_time],
        'stops': [stops],
        'arrival_time': [arrival_time],
        'destination_city': [destination_city],
        'class': [travel_class]
    })

    # Perform necessary preprocessing (encode categorical variables, etc.)
    # Note: Replace this with actual preprocessing steps used during training
    input_data = preprocess_input(input_data)

    # Make prediction
    prediction = model.predict(input_data)
    return f"Estimated Price: {prediction[0]:.2f}$"

# Define Gradio interface
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Dropdown(
            label="Airline",
            choices=[
                "Air Asia",
                "Air India",
                "GoAir",
                "Indigo",
                "Jet Airways",
                "Multiple carriers",
                "SpiceJet",
                "Vistara",
            ],
        ),
        gr.Dropdown(
            label="Source City",
            choices=[
                "Bangalore",
                "Chennai",
                "Delhi",
                "Hyderabad",
                "Kolkata",
                "Mumbai",
            ],
        ),
        gr.Dropdown(
            label="Departure Time",
            choices=[
                "Early_Morning",
                "Morning",
                "Afternoon",
                "Evening",
                "Night",
                "Late_Night",
            ],
        ),
        gr.Dropdown(
            label="Stops",
            choices=["non-stop", "1 stop", "2 stops", "3 stops"],
        ),
        gr.Dropdown(
            label="Arrival Time",
            choices=[
                "Early_Morning",
                "Morning",
                "Afternoon",
                "Evening",
                "Night",
                "Late_Night",
            ],
        ),
        gr.Dropdown(
            label="Destination City",
            choices=[
                "Bangalore",
                "Chennai",
                "Delhi",
                "Hyderabad",
                "Kolkata",
                "Mumbai",
            ],
        ),
        gr.Dropdown(
            label="Class",
            choices=["Economy", "Business"],
        ),
    ],
    outputs="text",
    title="Flight Price Prediction",
    description="Enter flight details to predict the price.",
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch(share=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the dataset
data = pd.read_csv(r"my own generate dataset.csv")

# Encode the 'Part Name' column
le = LabelEncoder()
data["Part Name"] = le.fit_transform(data["Part Name"])

# Prepare features (X) and target variable (y)
X = data.drop(columns=["Suggested Material"])  # Adjust "Suggested Material" to your actual target column name
y = data["Suggested Material"].values  # Adjust to your actual target column

# Save the column names for later use
column_names = X.columns

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2000)

# Initialize and train the Naive Bayes Classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", accuracy)

# Function to predict suggested material based on input features
def predict_suggested_material(part_name, density, tensile_strength, stiffness, thermal_conductivity, cost, stress_requirement, operating_temperature):
    # Create a DataFrame with all columns set to zero, matching the training data structure
    input_data = pd.DataFrame(np.zeros((1, len(column_names))), columns=column_names)
    
    # Set the input features to their respective values
    input_data["Density"] = density
    input_data["Tensile_Strength"] = tensile_strength
    input_data["Stiffness"] = stiffness
    input_data["Thermal_Conductivity"] = thermal_conductivity
    input_data["Cost"] = cost
    input_data["Stress_Requirement"] = stress_requirement
    input_data["Operating_Temperature"] = operating_temperature
    
    # Encode the 'Part Name' similarly to the training process
    if part_name in le.classes_:
        part_name_encoded = le.transform([part_name])[0]
    else:
        raise ValueError(f"Part Name '{part_name}' is not in the training data")
    input_data["Part Name"] = part_name_encoded
    
    # Ensure that the input data has the same columns as the training data
    input_data = input_data[column_names]
    
    # Make prediction using the DataFrame with feature names
    prediction = model.predict(input_data)
    
    return prediction[0]

# Gather user input with column names and units
part_name = input("Enter Part Name: ")
density = float(input("Enter Density (g/cm^3): "))  # Example unit for Density
tensile_strength = float(input("Enter Tensile Strength (MPa): "))  # Example unit for Tensile Strength
stiffness = float(input("Enter Stiffness (GPa): "))  # Example unit for Stiffness
thermal_conductivity = float(input("Enter Thermal Conductivity (W/mK): "))  # Example unit for Thermal Conductivity
cost = float(input("Enter Cost (RS): "))  # Cost in RS
stress_requirement = float(input("Enter Stress Requirement (MPa): "))  # Example unit for Stress Requirement
operating_temperature = float(input("Enter Operating Temperature (°C): "))  # Temperature in Degree Centigrade

# Make prediction based on user input
try:
    suggested_material = predict_suggested_material(part_name, density, tensile_strength, stiffness, thermal_conductivity, cost, stress_requirement, operating_temperature)
    print("Suggested Material:", suggested_material)
except ValueError as e:
    print(e)

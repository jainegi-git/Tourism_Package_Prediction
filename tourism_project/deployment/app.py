import streamlit as st
import pandas as pd
import joblib
import os

# Define the path to the model
model_path = "best_model.joblib" # Corrected path for deployment to HF Spaces

# Check if the model file exists, if not, download it (though it should be copied by Dockerfile)
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please ensure the model is copied to the deployment directory.")
else:
    # Load the model
    model = joblib.load(model_path)

    # Streamlit UI for Wellness Tourism Package Prediction
    st.title("Wellness Tourism Package Prediction App")
    st.write("""
    This application predicts whether a customer will purchase the Wellness Tourism Package
    based on their details and interaction data. Please enter the customer details below.
    """)

    # Define mappings for label encoded features (derived from bank_dataset during training)
    typeofcontact_mapping = {'Company Invited': 0, 'Self Inquiry': 1}
    occupation_mapping = {'Freelancer': 0, 'Large Business': 1, 'Salaried': 2, 'Small Business': 3, 'Unemployed': 4}
    gender_mapping = {'Female': 0, 'Male': 1}
    maritalstatus_mapping = {'Divorced': 0, 'Married': 1, 'Single': 2}
    designation_mapping = {'AVP': 0, 'Associate': 1, 'CEO': 2, 'Chairman': 3, 'Director': 4, 'Executive': 5, 'Junior Executive': 6, 'Manager': 7, 'President': 8, 'Senior Manager': 9, 'VP': 10}
    productpitched_values = ['Luxury', 'Deluxe', 'Standard', 'Basic', 'Super Deluxe']

    # User input for features
    st.header("Customer Details")
    age = st.number_input("Age", min_value=18, max_value=90, value=30)
    monthly_income = st.number_input("Monthly Income", min_value=0.0, value=25000.0, step=1000.0)
    number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1)
    number_of_children_visiting = st.number_input("Number of Children Visiting (below 5)", min_value=0, max_value=5, value=0)
    number_of_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5)
    passport = st.selectbox("Passport", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    own_car = st.selectbox("Own Car", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    city_tier = st.selectbox("City Tier", options=[1, 2, 3])
    preferred_property_star = st.selectbox("Preferred Property Star Rating", options=[3, 4, 5])

    st.header("Interaction Details")
    type_of_contact_str = st.selectbox("Type of Contact", options=list(typeofcontact_mapping.keys()))
    occupation_str = st.selectbox("Occupation", options=list(occupation_mapping.keys()))
    gender_str = st.selectbox("Gender", options=list(gender_mapping.keys()))
    marital_status_str = st.selectbox("Marital Status", options=list(maritalstatus_mapping.keys()))
    designation_str = st.selectbox("Designation", options=list(designation_mapping.keys()))
    product_pitched_str = st.selectbox("Product Pitched", options=productpitched_values)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=10.0, step=1.0)
    number_of_followups = st.number_input("Number of Follow-ups", min_value=0.0, value=2.0, step=1.0)
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)

    # Convert categorical inputs to numerical using mappings
    type_of_contact = typeofcontact_mapping[type_of_contact_str]
    occupation = occupation_mapping[occupation_str]
    gender = gender_mapping[gender_str]
    marital_status = maritalstatus_mapping[marital_status_str]
    designation = designation_mapping[designation_str]
    product_pitched = product_pitched_str # OneHotEncoder in pipeline handles this string

    # Assemble input into DataFrame
    input_data = pd.DataFrame([{
        'Age': age,
        'TypeofContact': type_of_contact,
        'CityTier': city_tier,
        'DurationOfPitch': duration_of_pitch,
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': number_of_person_visiting,
        'NumberOfFollowups': number_of_followups,
        'ProductPitched': product_pitched,
        'PreferredPropertyStar': preferred_property_star,
        'MaritalStatus': marital_status,
        'NumberOfTrips': number_of_trips,
        'Passport': passport,
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'OwnCar': own_car,
        'NumberOfChildrenVisiting': number_of_children_visiting,
        'Designation': designation,
        'MonthlyIncome': monthly_income
    }])

    if st.button("Predict Purchase"): # Renamed button text
        if 'model' in locals(): # Check if model was loaded successfully
            # Make prediction
            # The model pipeline handles preprocessing (scaling and one-hot encoding)
            prediction_proba = model.predict_proba(input_data)[:, 1]
            # Use the same classification_threshold as in training (0.55)
            classification_threshold = 0.55
            prediction = (prediction_proba >= classification_threshold).astype(int)[0]

            result_text = "Customer will purchase the Wellness Tourism Package" if prediction == 1 else "Customer will NOT purchase the Wellness Tourism Package"
            st.subheader("Prediction Result:")
            st.success(f"The model predicts: **{result_text}**")
            st.write(f"Purchase probability: {prediction_proba[0]:.2f}")
        else:
            st.error("Model could not be loaded. Cannot make prediction.")

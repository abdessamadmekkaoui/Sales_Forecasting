import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("📊 Sales Prediction App")
st.markdown("""
This app allows you to make sales predictions using trained machine learning models.
Enter the required features below, and the app will automatically calculate date-related features.
""")

# Load the saved models and label encoder
@st.cache_resource
def load_models():
    try:
        random_forest = joblib.load('random_forest_model.pkl')
        lightgbm = joblib.load('lightgbm_model.pkl')
        xgboost = joblib.load('xgboost_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        
        # Debug model information
        st.session_state['model_info'] = {
            'Random Forest': str(type(random_forest)),
            'LightGBM': str(type(lightgbm)),
            'XGBoost': str(type(xgboost))
        }
        
        return {
            'Random Forest': random_forest,
            'LightGBM': lightgbm,
            'XGBoost': xgboost
        }, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

models, label_encoder = load_models()

# Function to calculate date-derived features
def calculate_date_features(selected_date):
    date_features = {}
    
    # Extract basic date components
    date_features['Year'] = selected_date.year
    date_features['Month'] = selected_date.month
    date_features['Day'] = selected_date.day
    
    # Calculate additional date features
    date_features['DayOfWeek'] = selected_date.weekday()  # Monday=0, Sunday=6
    date_features['WeekOfYear'] = selected_date.isocalendar()[1]
    date_features['Quarter'] = (selected_date.month - 1) // 3 + 1
    date_features['IsWeekend'] = 1 if selected_date.weekday() >= 5 else 0
    
    # Month start/end
    date_features['IsMonthStart'] = 1 if selected_date.day == 1 else 0
    last_day = pd.Timestamp(selected_date.year, selected_date.month, 1) + pd.offsets.MonthEnd(1)
    date_features['IsMonthEnd'] = 1 if selected_date.day == last_day.day else 0
    
    # Days since start (assuming start date is Jan 1, 2020 - adjust as needed)
    start_date = datetime.date(2020, 1, 1)
    date_features['DaysSinceStart'] = (selected_date - start_date).days
    
    date_features['DayOfMonth'] = selected_date.day
    
    return date_features

# Create a sidebar for inputs
st.sidebar.header("Input Features")

# Branch selection
branch_options = ["Mellousa", "Laayoune", "Safsaf", "Lafayette", "Fes", 
                "Tahla", "Al Jazira", "Grande Route", "Doumer", "Manar",
                "El Menzeh", "Exit Casa", "Bir Jdid", "Benguerir", "Oum Rabii",
                "Amskroud", "Benguerir2", "Shell Select Etoile"]
selected_branch = st.sidebar.selectbox("Branch", branch_options)

# Date picker
selected_date = st.sidebar.date_input("Date", datetime.datetime.now())

# Boolean features with checkboxes
col1, col2 = st.sidebar.columns(2)
with col1:
    is_holiday = st.checkbox("Is Holiday", value=False)
    is_ramadan = st.checkbox("Is Ramadan", value=False)
with col2:
    is_eid = st.checkbox("Is Eid", value=False)
    is_school_vacation = st.checkbox("Is School Vacation", value=False)

# Weather and other features
temperature = st.sidebar.slider("Temperature (°C)", -10.0, 50.0, 25.0, 0.1)
precipitation = st.sidebar.slider("Precipitation (mm)", 0.0, 100.0, 0.0, 0.1)
extreme_weather = st.sidebar.checkbox("Extreme Weather", value=False)
is_payday = st.sidebar.checkbox("Is Payday", value=False)

# Prepare feature data for prediction
def prepare_input_data():
    # Get date-derived features
    date_features = calculate_date_features(selected_date)
    
    # Encode branch
    if label_encoder is not None:
        try:
            # Try to encode the branch name using the label encoder
            branch_idx = label_encoder.transform([selected_branch])[0]
        except:
            # If that fails, use the index as a fallback
            branch_idx = branch_options.index(selected_branch)
    else:
        branch_idx = branch_options.index(selected_branch)
    
    # Combine all features
    data = {
        'Branch': branch_idx,
        'Year': date_features['Year'],
        'Month': date_features['Month'],
        'Day': date_features['Day'],
        'DayOfWeek': date_features['DayOfWeek'],
        'WeekOfYear': date_features['WeekOfYear'],
        'Quarter': date_features['Quarter'],
        'IsWeekend': date_features['IsWeekend'],
        'IsMonthStart': date_features['IsMonthStart'],
        'IsMonthEnd': date_features['IsMonthEnd'],
        'IsHoliday': 1 if is_holiday else 0,
        'IsRamadan': 1 if is_ramadan else 0,
        'IsEid': 1 if is_eid else 0,
        'DaysSinceStart': date_features['DaysSinceStart'],
        'DayOfMonth': date_features['DayOfMonth'],
        'IsSchoolVacation': 1 if is_school_vacation else 0,
        'Temperature': temperature,
        'Precipitation': precipitation,
        'ExtremeWeather': 1 if extreme_weather else 0,
        'IsPayday': 1 if is_payday else 0
    }
    
    # Create DataFrame with features in the correct order
    features = [
        'Branch', 'Year', 'Month', 'Day', 
        'DayOfWeek', 'WeekOfYear', 'Quarter', 'IsWeekend', 
        'IsMonthStart', 'IsMonthEnd', 'IsHoliday', 'IsRamadan', 
        'IsEid', 'DaysSinceStart', 'DayOfMonth', 'IsSchoolVacation', 'Temperature', 'Precipitation',
        'ExtremeWeather', 'IsPayday'
    ]
    
    # Debug input data
    st.session_state['input_data'] = data
    
    df = pd.DataFrame([data], columns=features)
    return df

# Function to make predictions with selected models
def predict_sales(input_data):
    if models is None:
        st.error("Models not loaded properly. Please check model files.")
        return None
    
    results = {}
    for model_name, model in models.items():
        try:
            # Debug the input data
            st.session_state[f'input_shape_{model_name}'] = input_data.shape
            
            prediction = model.predict(input_data)[0]
            results[model_name] = prediction
            
            # Debug the prediction
            st.session_state[f'raw_prediction_{model_name}'] = prediction
        except Exception as e:
            st.error(f"Error predicting with {model_name}: {e}")
            results[model_name] = None
    
    return results

# Make predictions button
if st.sidebar.button("Predict Sales"):
    # Display input summary
    st.header("Input Summary")
    
    # Calculate date features for display
    date_features = calculate_date_features(selected_date)
    
    # Create two columns for the input summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User-Provided Features")
        user_features = {
            "Branch": selected_branch,
            "Date": selected_date.strftime("%Y-%m-%d"),
            "Is Holiday": "Yes" if is_holiday else "No",
            "Is Ramadan": "Yes" if is_ramadan else "No",
            "Is Eid": "Yes" if is_eid else "No",
            "Is School Vacation": "Yes" if is_school_vacation else "No",
            "Temperature (°C)": f"{temperature:.1f}",
            "Precipitation (mm)": f"{precipitation:.1f}",
            "Extreme Weather": "Yes" if extreme_weather else "No",
            "Is Payday": "Yes" if is_payday else "No"
        }
        st.table(pd.DataFrame(user_features.items(), columns=["Feature", "Value"]))
    
    with col2:
        st.subheader("Derived Date Features")
        derived_features = {
            "Year": date_features["Year"],
            "Month": date_features["Month"],
            "Day": date_features["Day"],
            "Day of Week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][date_features["DayOfWeek"]],
            "Week of Year": date_features["WeekOfYear"],
            "Quarter": date_features["Quarter"],
            "Is Weekend": "Yes" if date_features["IsWeekend"] else "No",
            "Is Month Start": "Yes" if date_features["IsMonthStart"] else "No",
            "Is Month End": "Yes" if date_features["IsMonthEnd"] else "No",
            "Day of Month": date_features["DayOfMonth"],
            "Days Since Start": date_features["DaysSinceStart"]
        }
        st.table(pd.DataFrame(derived_features.items(), columns=["Feature", "Value"]))
    
    # Prepare input data and make predictions
    input_data = prepare_input_data()
    predictions = predict_sales(input_data)
    
    # Debug section
    with st.expander("Debug Information (click to expand)"):
        st.subheader("Model Types")
        if 'model_info' in st.session_state:
            st.write(st.session_state['model_info'])
        
        st.subheader("Input Data")
        if 'input_data' in st.session_state:
            st.write(st.session_state['input_data'])
        
        st.subheader("Input Shape")
        for model_name in models.keys():
            key = f'input_shape_{model_name}'
            if key in st.session_state:
                st.write(f"{model_name} input shape: {st.session_state[key]}")
        
        st.subheader("Raw Predictions")
        for model_name in models.keys():
            key = f'raw_prediction_{model_name}'
            if key in st.session_state:
                st.write(f"{model_name} raw prediction: {st.session_state[key]}")
    
    if predictions:
        st.header("Sales Predictions")
        
        # Display predictions in a table
        pred_df = pd.DataFrame({
            "Model": list(predictions.keys()),
            "Predicted Sales": [f"{pred:.2f}" if pred is not None else "Error" for pred in predictions.values()]
        })
        st.table(pred_df)
        
        # Calculate average prediction
        valid_predictions = [v for v in predictions.values() if v is not None]
        if valid_predictions:
            avg_prediction = sum(valid_predictions) / len(valid_predictions)
            st.success(f"Average prediction across all models: {avg_prediction:.2f}")
        
        # Bar chart for predictions
        fig, ax = plt.subplots(figsize=(10, 5))
        valid_predictions_dict = {k: v for k, v in predictions.items() if v is not None}
        if valid_predictions_dict:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            bars = plt.bar(valid_predictions_dict.keys(), valid_predictions_dict.values(), color=colors)
            plt.title('Sales Predictions by Model')
            plt.ylabel('Predicted Sales')
            plt.xlabel('Model')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{height:.2f}',
                        ha='center', va='bottom', rotation=0)
            
            st.pyplot(fig)
        else:
            st.error("No valid predictions to display")
        
        # Model comparison section
        st.header("Models Comparison")
        st.write("""
        This section shows the relative confidence and accuracy of each model.
        In a production environment, you might want to consider an ensemble approach
        or rely on the model with historically better performance.
        """)
        
        # Model info
        model_info = {
            "Random Forest": "Robust to overfitting, handles nonlinear relationships well",
            "LightGBM": "Gradient boosting framework, efficient with large datasets",
            "XGBoost": "Gradient boosting algorithm, high performance and speed"
        }
        
        for model_name, description in model_info.items():
            st.subheader(f"{model_name}")
            st.write(description)
            if model_name in predictions and predictions[model_name] is not None:
                st.success(f"Prediction: {predictions[model_name]:.2f}")
            else:
                st.error("Prediction failed")

# Add a section about the features
with st.expander("About the Features"):
    st.write("""
    ### Feature Descriptions:
    
    #### User Input Features:
    - **Branch**: The store location
    - **Date**: Selected date for prediction
    - **Is Holiday**: Whether the selected date is a public holiday
    - **Is Ramadan**: Whether the selected date falls during Ramadan
    - **Is Eid**: Whether the selected date is during Eid celebration
    - **Is School Vacation**: Whether schools are on vacation
    - **Temperature**: Temperature in Celsius
    - **Precipitation**: Rainfall amount in mm
    - **Extreme Weather**: Whether there's extreme weather (storm, heatwave, etc.)
    - **Is Payday**: Whether the date is a typical payday
    
    #### Automatically Calculated Features:
    - **Day of Week**: The day of the week (Monday to Sunday)
    - **Week of Year**: Which week of the year the date falls in
    - **Quarter**: Business quarter (1-4)
    - **Is Weekend**: Whether the date falls on a weekend
    - **Is Month Start/End**: Whether the date is at the start or end of a month
    - **Days Since Start**: Number of days since a reference start date
    - **Day of Month**: The day number within the month
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses machine learning models to predict sales based on various features. "
    "The models were trained using historical sales data and various environmental factors."
)
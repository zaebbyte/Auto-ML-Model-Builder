import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle

# Scikit-Learn imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

# --- Configuration ---
st.set_page_config(page_title="No-Code AutoML", layout="wide", page_icon="🤖")

# Custom CSS to make blocks look like a visual programming canvas
st.markdown("""
    <style>
    .block-container {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 2rem;
        border-left: 5px solid #4CAF50;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .instructor-panel {
        padding: 2rem;
        border-radius: 10px;
        background-color: #e8f4f8;
        border: 2px solid #2196F3;
        height: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'step' not in st.session_state:
    st.session_state['step'] = 1
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'target_col' not in st.session_state:
    st.session_state['target_col'] = None
if 'models_results' not in st.session_state:
    st.session_state['models_results'] = None
if 'best_model' not in st.session_state:
    st.session_state['best_model'] = None
if 'best_model_name' not in st.session_state:
    st.session_state['best_model_name'] = None
if 'task_type' not in st.session_state:
    st.session_state['task_type'] = None
if 'feature_importances' not in st.session_state:
    st.session_state['feature_importances'] = None
if 'feature_names' not in st.session_state:
    st.session_state['feature_names'] = None

# --- Main Layout ---
col_main, col_instructor = st.columns([3, 1])

# --- Helper Functions ---
def get_task_type(series):
    # Heuristic to determine if task is classification or regression
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() < 20: # Arbitrary threshold for classification
            return 'Classification'
        else:
            return 'Regression'
    else:
        return 'Classification'

def build_and_train_pipeline(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    task_type = get_task_type(y)
    st.session_state['task_type'] = task_type
    
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models based on task type
    if task_type == 'Classification':
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
    else:
        models = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'Linear Regression': LinearRegression()
        }
        
    results = []
    best_score = -np.inf
    best_model = None
    best_model_name = None
    feature_importances = None
    feature_names = None
    
    for name, model in models.items():
        # Create full pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        if task_type == 'Classification':
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            score = acc # Use accuracy for model selection
            results.append({'Model': name, 'Accuracy': acc, 'F1 Score': f1})
        else:
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            score = r2 # Use R2 for model selection
            results.append({'Model': name, 'R2 Score': r2, 'RMSE': rmse})
            
        if score > best_score:
            best_score = score
            best_model = pipeline
            best_model_name = name
            
            # Extract feature importances if possible (e.g. Random Forest)
            if hasattr(model, 'feature_importances_'):
                # Try to get feature names after preprocessing
                try:
                    # Get feature names from one hot encoder if present
                    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                    cat_features_encoded = cat_encoder.get_feature_names_out(categorical_features)
                    all_features = numeric_features + list(cat_features_encoded)
                    
                    feature_names = all_features
                    feature_importances = model.feature_importances_
                except Exception as e:
                    pass

    return pd.DataFrame(results), best_model, best_model_name, feature_importances, feature_names


with col_main:
    st.title("🧩 No-Code AutoML Platform")
    
    # --- Step 1: Upload CSV ---
    st.markdown('<div class="block-container">', unsafe_allow_html=True)
    st.subheader("Step 1: Upload CSV")
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("The uploaded CSV file is empty. Please upload a valid dataset.")
            else:
                st.session_state['df'] = df
                if st.session_state['step'] < 2:
                    st.session_state['step'] = 2
                st.success(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns.")
                with st.expander("Preview Data"):
                    st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Step 2: Select Target ---
    if st.session_state['step'] >= 2:
        st.markdown('<div class="block-container">', unsafe_allow_html=True)
        st.subheader("Step 2: Select Target")
        
        columns = st.session_state['df'].columns.tolist()
        # Ensure previously selected target is still valid
        idx = 0
        if st.session_state['target_col'] in columns:
            idx = columns.index(st.session_state['target_col'])
            
        target = st.selectbox("Select the column you want to predict:", columns, index=idx)
        
        if target:
            if st.session_state['target_col'] != target:
                 # Reset state if target changes
                 st.session_state['target_col'] = target
                 st.session_state['models_results'] = None
                 st.session_state['best_model'] = None
                 st.session_state['step'] = 3
            elif st.session_state['step'] < 3:
                 st.session_state['step'] = 3
                 
        st.markdown('</div>', unsafe_allow_html=True)
        
    # --- Step 3: Train Model ---
    if st.session_state['step'] >= 3:
        st.markdown('<div class="block-container">', unsafe_allow_html=True)
        st.subheader("Step 3: Train Model")
        
        if st.button("🚀 Run AutoML Pipeline", type="primary"):
            with st.spinner("Training models... This might take a moment."):
                df = st.session_state['df']
                target = st.session_state['target_col']
                
                try:
                    results_df, best_model, best_name, importances, feat_names = build_and_train_pipeline(df, target)
                    
                    st.session_state['models_results'] = results_df
                    st.session_state['best_model'] = best_model
                    st.session_state['best_model_name'] = best_name
                    st.session_state['feature_importances'] = importances
                    st.session_state['feature_names'] = feat_names
                    
                    st.session_state['step'] = 4
                    st.success("AutoML Pipeline completed successfully!")
                except Exception as e:
                    st.error(f"An error occurred during training: {e}")
                    
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Step 4: Results & Export ---
    if st.session_state['step'] >= 4 and st.session_state['models_results'] is not None:
        st.markdown('<div class="block-container">', unsafe_allow_html=True)
        st.subheader("Step 4: Visualize & Download")
        
        task = st.session_state['task_type']
        st.write(f"**Detected Task Type:** {task}")
        
        st.write("### Model Leaderboard")
        subset_cols = ['Accuracy', 'F1 Score'] if task == 'Classification' else ['R2 Score']
        # Remove columns that don't exist
        subset_cols = [c for c in subset_cols if c in st.session_state['models_results'].columns]
        
        st.dataframe(st.session_state['models_results'].style.highlight_max(axis=0, color='lightgreen', subset=subset_cols))
        
        st.write(f"**Best Model:** 🏆 {st.session_state['best_model_name']}")
        
        # Feature Importance Chart
        if st.session_state['feature_importances'] is not None and st.session_state['feature_names'] is not None:
            st.write("### Feature Importance")
            # Create a dataframe for the chart
            fi_df = pd.DataFrame({
                'Feature': st.session_state['feature_names'],
                'Importance': st.session_state['feature_importances']
            }).sort_values(by='Importance', ascending=False).head(10) # Top 10 features
            
            # Set 'Feature' as index for st.bar_chart
            fi_df.set_index('Feature', inplace=True)
            st.bar_chart(fi_df)
        else:
            st.info("Feature importance is not available for the best model (e.g., Linear/Logistic Regression without coefficients extracted or due to pipeline complexity).")
        
        # Download Best Model
        st.write("### Export Best Model")
        # Serialize model
        model_bytes = pickle.dumps(st.session_state['best_model'])
        
        st.download_button(
            label="Download Best Model (.pkl)",
            data=model_bytes,
            file_name="best_automl_model.pkl",
            mime="application/octet-stream"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)


# --- Instructor Panel ---
with col_instructor:
    st.markdown('<div class="instructor-panel">', unsafe_allow_html=True)
    st.subheader("👩‍🏫 Instructor")
    
    step = st.session_state['step']
    
    if step == 1:
        st.write("Welcome to the No-Code AutoML Platform!")
        st.write("To get started, please **upload a CSV file** containing your dataset in the Main Workspace on the left.")
        st.write("Make sure your dataset has headers and is nicely formatted.")
        
    elif step == 2:
        df = st.session_state['df']
        st.write(f"Great! I see your data has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
        st.write("Next, **select what you want to predict** (your Target Variable) from the dropdown.")
        st.write("I will automatically figure out if it's a Classification or Regression problem.")
        
    elif step == 3:
        target = st.session_state['target_col']
        st.write(f"Awesome! You selected **`{target}`** as your target.")
        st.write("Now, simply click **'🚀 Run AutoML Pipeline'** to let me do the hard work.")
        st.write("I will clean the data, handle missing values, encode text, and try out a few different algorithms.")
        
    elif step >= 4:
        task = st.session_state['task_type']
        best_name = st.session_state['best_model_name']
        st.write("🎉 **Training Complete!**")
        st.write(f"I determined this was a **{task}** task.")
        st.write(f"The best performing model was **{best_name}**.")
        st.write("Check out the leaderboard and feature importances to see what drives your data.")
        st.write("Finally, don't forget to **download your model** so you can use it in production!")
        
        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

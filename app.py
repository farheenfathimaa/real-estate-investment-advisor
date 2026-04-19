import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import mlflow

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

def predict_good_investment(cls_model, X):
    if hasattr(cls_model, 'predict_proba'):
        prob = cls_model.predict_proba(X)[0][1]
        pred = 1 if prob >= 0.5 else 0
        return pred, prob
    else:
        pred = cls_model.predict(X)[0]
        return pred, 1.0 if pred == 1 else 0.0

@st.cache_resource(show_spinner=False)
def load_models_and_transformers():
    cls_path = "models/best_classifier.pkl"
    reg_path = "models/best_regressor.pkl"
    le_path = "models/label_encoders.pkl"
    scaler_path = "models/scaler.pkl"
    feat_path = "models/feature_columns.pkl"
    
    if not all(os.path.exists(p) for p in [cls_path, reg_path, le_path, scaler_path, feat_path]):
        return None, None, None, None, None
        
    classifier = joblib.load(cls_path)
    regressor = joblib.load(reg_path)
    encoders = joblib.load(le_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(feat_path)
    # also attempt to load scaled cols tracking
    scaled_cols_path = "models/scaled_columns.pkl"
    scaled_cols = joblib.load(scaled_cols_path) if os.path.exists(scaled_cols_path) else []
    
    return classifier, regressor, encoders, scaler, features, scaled_cols

# Uncached data load for fresh EDA filtering
def load_raw_data(encoders, scaler, scaled_cols):
    if not os.path.exists("cleaned_data.csv"):
        return None
    df = pd.read_csv("cleaned_data.csv")
    
    # inverse transform numericals
    if scaled_cols:
        existing_cols = [c for c in scaled_cols if c in df.columns]
        df[existing_cols] = scaler.inverse_transform(df[existing_cols])
        
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.inverse_transform(df[col].astype(int))
            
    return df

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["1. Property Investment Predictor", "2. EDA Dashboard", "3. Model Performance"])
    
    models = load_models_and_transformers()
    if models[0] is None:
        st.error("Models not found. Please train models first using `train_models.py`.")
        return
        
    classifier, regressor, encoders, scaler, features, scaled_cols = models
    df_raw = load_raw_data(encoders, scaler, scaled_cols)
    
    if page == "1. Property Investment Predictor":
        st.title("🏡 Property Investment Predictor")
        st.write("Input property details to predict if it's a good investment and estimate its future 5-year price.")
        
        # Build forms with dynamic dropdowns from the unencoded dataset
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                state = st.selectbox("State", encoders['State'].classes_ if 'State' in encoders else ["State"])
                city = st.selectbox("City", encoders['City'].classes_ if 'City' in encoders else ["City"])
                property_type = st.selectbox("Property Type", encoders['Property_Type'].classes_ if 'Property_Type' in encoders else ["Villa", "Apartment", "Independent House"])
                bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
                size_sqft = st.number_input("Size (SqFt)", min_value=100, max_value=50000, value=1500)
                price_lakhs = st.number_input("Current Price (in Lakhs)", min_value=1.0, max_value=5000.0, value=50.0)
                year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2015)
                
            with col2:
                furnished = st.selectbox("Furnished Status", encoders['Furnished_Status'].classes_ if 'Furnished_Status' in encoders else ["Unfurnished", "Semi-furnished", "Furnished"])
                floor_no = st.number_input("Floor No", min_value=0, max_value=100, value=2)
                total_floors = st.number_input("Total Floors", min_value=1, max_value=100, value=5)
                nearby_schools = st.number_input("Nearby Schools (count)", min_value=0, max_value=20, value=2)
                nearby_hospitals = st.number_input("Nearby Hospitals (count)", min_value=0, max_value=20, value=1)
                facing = st.selectbox("Facing", encoders['Facing'].classes_ if 'Facing' in encoders else ["North", "South", "East", "West"])
                
            with col3:
                pub_trans = st.selectbox("Public Transport Accessibility", encoders['Public_Transport_Accessibility'].classes_ if 'Public_Transport_Accessibility' in encoders else ["Low", "Medium", "High"])
                parking = st.selectbox("Parking Space", encoders['Parking_Space'].classes_ if 'Parking_Space' in encoders else ["No", "Yes"])
                security = st.selectbox("Security", encoders['Security'].classes_ if 'Security' in encoders else ["No", "Yes"])
                amenity = st.selectbox("Amenities Package", encoders['Amenities'].classes_[:10] if 'Amenities' in encoders else ["Pool", "Gym"])
                owner_type = st.selectbox("Owner Type", encoders['Owner_Type'].classes_ if 'Owner_Type' in encoders else ["Builder", "Owner", "Broker"])
                availability = st.selectbox("Availability", encoders['Availability_Status'].classes_ if 'Availability_Status' in encoders else ["Ready_to_Move", "Under_Construction"])
                
            submitted = st.form_submit_button("Predict")
            
        if submitted:
            # 1. Collect inputs into a dict
            input_data = {
                'State': state, 'City': city, 'Property_Type': property_type, 'BHK': bhk, 
                'Size_in_SqFt': size_sqft, 'Price_in_Lakhs': price_lakhs, 'Year_Built': year_built,
                'Furnished_Status': furnished, 'Floor_No': floor_no, 'Total_Floors': total_floors,
                'Nearby_Schools': nearby_schools, 'Nearby_Hospitals': nearby_hospitals,
                'Public_Transport_Accessibility': pub_trans, 'Parking_Space': parking,
                'Security': security, 'Amenities': amenity, 'Facing': facing,
                'Owner_Type': owner_type, 'Availability_Status': availability
            }
            
            # Note: We need 'Locality'. Since it's huge, we'll dummy code it with the mode or first class
            if 'Locality' not in input_data and 'Locality' in encoders:
                input_data['Locality'] = encoders['Locality'].classes_[0]
                
            df_input = pd.DataFrame([input_data])
            
            # Engineered features
            df_input['Price_per_SqFt'] = df_input['Price_in_Lakhs'] / df_input['Size_in_SqFt']
            df_input['Age_of_Property'] = 2025 - df_input['Year_Built']
            # Fake min-max for school density (assuming min 0, max 10 based on our manual look)
            df_input['School_Density_Score'] = df_input['Nearby_Schools'] / 10.0
            
            # Encode categoricals safely
            for col, le in encoders.items():
                if col in df_input.columns:
                    val = df_input[col].iloc[0]
                    # Handle unseen nicely
                    if val in le.classes_:
                        df_input[col] = le.transform([val])[0]
                    else:
                        df_input[col] = 0 # Default to 0 index if unseen
                        
            # Scale numericals
            existing_scaled_cols = [c for c in scaled_cols if c in df_input.columns]
            if existing_scaled_cols:
                # To transform a single row without shape issues, we need to pass a slice with exactly the columns the scaler expects
                # BUT scaler expects ALL columns it was fit on. We construct a 1-row DataFrame holding all scaled_cols in correct order
                temp_scale = pd.DataFrame(np.zeros((1, len(scaled_cols))), columns=scaled_cols)
                for c in existing_scaled_cols:
                    temp_scale[c] = df_input[c]
                scaled_res = scaler.transform(temp_scale)
                df_input[existing_scaled_cols] = scaled_res[0, [scaled_cols.index(c) for c in existing_scaled_cols]]
                
            # Align features
            missing_cols = [c for c in features if c not in df_input.columns]
            for c in missing_cols:
                df_input[c] = 0 # Dummy fill for missing
            X = df_input[features]
            
            # Predict Good Investment
            pred_inv, prob_inv = predict_good_investment(classifier, X)
            
            # Predict Future Price
            pred_price = regressor.predict(X)[0]
            
            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.subheader("Classification Outcome")
                if pred_inv == 1:
                    st.success(f"✅ Good Investment (Confidence: {prob_inv*100:.1f}%)")
                else:
                    st.error(f"❌ Not a Good Investment (Confidence: {(1-prob_inv)*100:.1f}%)")
            with res_col2:
                st.subheader("Regression Outcome")
                st.info(f"💰 Estimated Price after 5 Years:\n\n**₹ {pred_price:.2f} Lakhs**")
                
            # Feature Importance
            st.markdown("### Decision Drivers (Feature Importance)")
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                indices = np.argsort(importances)[-10:]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(range(10), importances[indices], align='center', color='indigo')
                ax.set_yticks(range(10))
                ax.set_yticklabels(np.array(features)[indices])
                st.pyplot(fig)
            elif hasattr(classifier, 'coef_'):
                importances = np.abs(classifier.coef_[0])
                indices = np.argsort(importances)[-10:]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(range(10), importances[indices], align='center', color='indigo')
                ax.set_yticks(range(10))
                ax.set_yticklabels(np.array(features)[indices])
                st.pyplot(fig)
                
    elif page == "2. EDA Dashboard":
        st.title("📊 Exploratory Data Analysis Dashboard")
        st.write("View the 20 pre-generated EDA charts, or apply filters below to dynamically recreate them.")
        
        if df_raw is not None:
            col1, col2, col3 = st.columns(3)
            filter_state = col1.selectbox("Filter by State", ["All"] + sorted(list(df_raw['State'].dropna().unique())))
            filter_city = col2.selectbox("Filter by City", ["All"] + sorted(list(df_raw['City'].dropna().unique())))
            filter_prop = col3.selectbox("Filter by Property Type", ["All"] + sorted(list(df_raw['Property_Type'].dropna().unique())))
            
            # Use raw dataset
            plot_df = df_raw.copy()
            filtered = False
            if filter_state != "All":
                plot_df = plot_df[plot_df['State'] == filter_state]
                filtered = True
            if filter_city != "All":
                plot_df = plot_df[plot_df['City'] == filter_city]
                filtered = True
            if filter_prop != "All":
                plot_df = plot_df[plot_df['Property_Type'] == filter_prop]
                filtered = True
                
            if filtered:
                st.warning("Filters applied! Rendering a subset of dynamic charts to save computation time.")
                if len(plot_df) == 0:
                    st.error("No data matches the selected filters.")
                else:
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        fig1, ax1 = plt.subplots()
                        sns.histplot(data=plot_df, x='Price_in_Lakhs', ax=ax1, color='blue')
                        ax1.set_title("Property Prices")
                        st.pyplot(fig1)

                        fig3, ax3 = plt.subplots()
                        avg_price_city = plot_df.groupby('City')['Price_in_Lakhs'].mean().nlargest(15)
                        sns.barplot(x=avg_price_city.index, y=avg_price_city.values, ax=ax3, palette='magma')
                        ax3.set_title("Avg Property Price by City")
                        ax3.tick_params(axis='x', rotation=45)
                        st.pyplot(fig3)
                        
                    with dc2:
                        fig2, ax2 = plt.subplots()
                        sns.scatterplot(data=plot_df, x='Size_in_SqFt', y='Price_in_Lakhs', alpha=0.5, ax=ax2)
                        ax2.set_title("Size vs Price")
                        st.pyplot(fig2)
                        
                        fig4, ax4 = plt.subplots()
                        sns.boxplot(data=plot_df, x='Furnished_Status', y='Price_in_Lakhs', ax=ax4, palette='Set3')
                        ax4.set_title("Price by Furnished Status")
                        st.pyplot(fig4)
            else:
                st.success("Showing Pre-generated Full-Dataset EDA Charts from /eda_charts/")
                charts = sorted(os.listdir("eda_charts"))
                # Filter to PNGs only
                charts = [c for c in charts if c.endswith(".png")]
                
                # Show in grid of 2
                for i in range(0, len(charts), 2):
                    i_col1, i_col2 = st.columns(2)
                    with i_col1:
                        st.image(os.path.join("eda_charts", charts[i]), use_column_width=True)
                    with i_col2:
                        if i+1 < len(charts):
                            st.image(os.path.join("eda_charts", charts[i+1]), use_column_width=True)
                        
    elif page == "3. Model Performance":
        st.title("📈 Model Performance Metrics")
        
        # Load mlflow runs
        try:
            exp = mlflow.get_experiment_by_name("RealEstate_Investment_Advisor")
            if exp:
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                if len(runs) > 0:
                    st.subheader("Classification Models Comparison")
                    # Filter for classification tracking values
                    cls_runs = runs[runs['metrics.F1_Score'].notnull()]
                    if len(cls_runs) > 0:
                        tbl_cls = cls_runs[['tags.mlflow.runName', 'metrics.Accuracy', 'metrics.Precision', 'metrics.Recall', 'metrics.F1_Score', 'metrics.ROC_AUC']]
                        tbl_cls = tbl_cls.rename(columns={'tags.mlflow.runName':'Model'}).drop_duplicates(subset=['Model']).set_index("Model")
                        st.dataframe(tbl_cls.style.highlight_max(axis=0, color='lightgreen'))
                
                    st.subheader("Regression Models Comparison")
                    reg_runs = runs[runs['metrics.RMSE'].notnull()]
                    if len(reg_runs) > 0:
                        tbl_reg = reg_runs[['tags.mlflow.runName', 'metrics.RMSE', 'metrics.MAE', 'metrics.R2']]
                        tbl_reg = tbl_reg.rename(columns={'tags.mlflow.runName':'Model'}).drop_duplicates(subset=['Model']).set_index("Model")
                        # RMSE/MAE lower is better, R2 higher is better
                        st.dataframe(tbl_reg.style.highlight_min(subset=['metrics.RMSE', 'metrics.MAE'], color='lightgreen').highlight_max(subset=['metrics.R2'], color='lightgreen'))
                        
            else:
                st.warning("MLFlow experiment runs not found yet.")
        except Exception as e:
            st.error(f"Could not load MLflow data: {e}")
            
        st.subheader("Confusion Matrix (Best Classifier)")
        # Show all confusion matrix pngs found in models folder
        cm_files = [f for f in os.listdir("models") if f.endswith("_cm.png")]
        if cm_files:
            c1, c2, c3 = st.columns(3)
            cols = [c1, c2, c3]
            for idx, c_file in enumerate(cm_files[:3]):
                with cols[idx]:
                    st.image(f"models/{c_file}", caption=c_file.replace('_cm.png', '').replace('_', ' '), use_column_width=True)

if __name__ == "__main__":
    main()

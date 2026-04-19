import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

def main():
    print("Loading data...")
    try:
        df = pd.read_csv("india_housing_prices.csv")
    except FileNotFoundError:
        print("Error: india_housing_prices.csv not found.")
        return
        
    print("Dropping duplicates...")
    df = df.drop_duplicates()
    
    print("Imputing missing values...")
    # Separate numeric and categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # Fill numeric with median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Fill categorical with mode
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    print("Engineering features...")
    if 'Price_per_SqFt' not in df.columns or df['Price_per_SqFt'].isnull().any():
        df['Price_per_SqFt'] = df['Price_in_Lakhs'] / df['Size_in_SqFt']
        
    df['Age_of_Property'] = 2025 - df['Year_Built']
    
    mms = MinMaxScaler()
    df['School_Density_Score'] = mms.fit_transform(df[['Nearby_Schools']])
    
    df['Future_Price_5Y'] = df['Price_in_Lakhs'] * (1.08 ** 5)
    
    median_price_sqft = df['Price_per_SqFt'].median()
    df['Good_Investment'] = ((df['Price_per_SqFt'] <= median_price_sqft) & (df['BHK'] >= 2)).astype(int)
    
    print("Encoding categorical features...")
    cat_cols_to_encode = ['State', 'City', 'Locality', 'Property_Type', 'Furnished_Status', 
                          'Security', 'Amenities', 'Facing', 'Owner_Type', 'Availability_Status',
                          'Parking_Space', 'Public_Transport_Accessibility']
    
    label_encoders = {}
    for col in cat_cols_to_encode:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            
    print("Scaling numerical features...")
    # Scale numerical predictors
    num_cols_to_scale = ['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt', 
                         'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property', 
                         'Nearby_Schools', 'Nearby_Hospitals', 'School_Density_Score',
                         'Parking_Space', 'Public_Transport_Accessibility']
    
    # Filter only existing numerical columns
    num_cols_to_scale = [c for c in num_cols_to_scale if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    scaler = StandardScaler()
    if num_cols_to_scale:
        df[num_cols_to_scale] = scaler.fit_transform(df[num_cols_to_scale])
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    print("Saving artifacts to models/...")
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(num_cols_to_scale, "models/scaled_columns.pkl") 
    
    feature_columns = [col for col in df.columns if col not in ['ID', 'Future_Price_5Y', 'Good_Investment']]
    joblib.dump(feature_columns, "models/feature_columns.pkl")
    
    print("Saving cleaned_data.csv...")
    df.to_csv("cleaned_data.csv", index=False)
    print("Preprocessing successfully completed.")

if __name__ == "__main__":
    main()

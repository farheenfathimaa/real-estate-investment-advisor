import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def main():
    print("Loading preprocessing artifacts and data...")
    # Read the data
    df = pd.read_csv("cleaned_data.csv")
    
    # Load transformers
    label_encoders = joblib.load("models/label_encoders.pkl")
    scaler = joblib.load("models/scaler.pkl")
    scaled_cols = joblib.load("models/scaled_columns.pkl")
    
    print("Inverse transforming data for readable EDA plots...")
    # Inverse transform numericals
    if scaled_cols:
        df[scaled_cols] = scaler.inverse_transform(df[scaled_cols])
        
    # Inverse transform categoricals
    for col, le in label_encoders.items():
        if col in df.columns:
            # We map strings through the integer cast
            df[col] = le.inverse_transform(df[col].astype(int))
            
    # Create eda directories
    os.makedirs("eda_charts", exist_ok=True)
    
    # Set seaborn style
    sns.set_theme(style="whitegrid")
    
    print("Generating EDA charts...")

    # 1. Distribution of property prices
    plt.figure(figsize=(10,6))
    sns.histplot(data=df, x='Price_in_Lakhs', bins=50, kde=True, color='blue')
    plt.title("Distribution of Property Prices (in Lakhs)")
    plt.tight_layout()
    plt.savefig("eda_charts/1_distribution_property_prices.png")
    plt.close()
    
    # 2. Distribution of property sizes
    plt.figure(figsize=(10,6))
    sns.histplot(data=df, x='Size_in_SqFt', bins=50, kde=True, color='green')
    plt.title("Distribution of Property Sizes (SqFt)")
    plt.tight_layout()
    plt.savefig("eda_charts/2_distribution_property_sizes.png")
    plt.close()
    
    # 3. Price per sq ft by property type (boxplot)
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x='Property_Type', y='Price_per_SqFt', palette='Set2')
    plt.title("Price per SqFt by Property Type")
    plt.tight_layout()
    plt.savefig("eda_charts/3_price_per_sqft_by_property_type.png")
    plt.close()

    # 4. Scatter: Size vs Price
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='Size_in_SqFt', y='Price_in_Lakhs', alpha=0.5)
    plt.title("Scatter Plot: Size vs Price")
    plt.tight_layout()
    plt.savefig("eda_charts/4_scatter_size_vs_price.png")
    plt.close()
    
    # 5. Outlier detection: boxplots for Price_per_SqFt and Size_in_SqFt
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    sns.boxplot(data=df, y='Price_per_SqFt', ax=axes[0], color='c')
    axes[0].set_title("Outliers: Price per SqFt")
    sns.boxplot(data=df, y='Size_in_SqFt', ax=axes[1], color='orange')
    axes[1].set_title("Outliers: Size in SqFt")
    plt.tight_layout()
    plt.savefig("eda_charts/5_outlier_detection.png")
    plt.close()
    
    # 6. Average price per sq ft by state (bar chart)
    plt.figure(figsize=(12,6))
    avg_price_state = df.groupby('State')['Price_per_SqFt'].mean().sort_values(ascending=False)
    sns.barplot(x=avg_price_state.index, y=avg_price_state.values, palette='viridis')
    plt.title("Average Price per Sq Ft by State")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("eda_charts/6_avg_price_per_sqft_by_state.png")
    plt.close()
    
    # 7. Average property price by city (top 15)
    plt.figure(figsize=(12,6))
    avg_price_city = df.groupby('City')['Price_in_Lakhs'].mean().nlargest(15)
    sns.barplot(x=avg_price_city.index, y=avg_price_city.values, palette='magma')
    plt.title("Average Property Price by City (Top 15)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("eda_charts/7_avg_property_price_top_15_cities.png")
    plt.close()
    
    # 8. Median age of properties by locality (top 15)
    plt.figure(figsize=(12,6))
    median_age_loc = df.groupby('Locality')['Age_of_Property'].median().nlargest(15)
    sns.barplot(x=median_age_loc.index, y=median_age_loc.values, palette='coolwarm')
    plt.title("Median Age of Properties by Locality (Top 15)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("eda_charts/8_median_age_top_15_localities.png")
    plt.close()
    
    # 9. BHK distribution across cities (heatmap or grouped bar)
    plt.figure(figsize=(12,8))
    top_10_cities = df['City'].value_counts().nlargest(10).index
    bhk_city_xtab = pd.crosstab(df[df['City'].isin(top_10_cities)]['City'], df['BHK'])
    sns.heatmap(bhk_city_xtab, annot=True, fmt='d', cmap='YlGnBu')
    plt.title("BHK Distribution Across Top 10 Cities")
    plt.tight_layout()
    plt.savefig("eda_charts/9_bhk_distribution_across_cities.png")
    plt.close()
    
    # 10. Price trends for top 5 most expensive localities
    plt.figure(figsize=(12,6))
    top5_loc = df.groupby('Locality')['Price_in_Lakhs'].mean().nlargest(5).index
    sns.lineplot(data=df[df['Locality'].isin(top5_loc)], x='Year_Built', y='Price_in_Lakhs', hue='Locality', ci=None)
    plt.title("Price Trends for Top 5 Most Expensive Localities")
    plt.tight_layout()
    plt.savefig("eda_charts/10_price_trends_top_5_localities.png")
    plt.close()
    
    # 11. Correlation heatmap of all numeric features
    plt.figure(figsize=(14,10))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), cmap='vlag', annot=False)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig("eda_charts/11_correlation_heatmap.png")
    plt.close()
    
    # 12. Nearby Schools vs Price_per_SqFt (scatter)
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='Nearby_Schools', y='Price_per_SqFt', alpha=0.5, color='purple')
    plt.title("Nearby Schools vs Price per SqFt")
    plt.tight_layout()
    plt.savefig("eda_charts/12_nearby_schools_vs_price_per_sqft.png")
    plt.close()
    
    # 13. Nearby Hospitals vs Price_per_SqFt (scatter)
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='Nearby_Hospitals', y='Price_per_SqFt', alpha=0.5, color='red')
    plt.title("Nearby Hospitals vs Price per SqFt")
    plt.tight_layout()
    plt.savefig("eda_charts/13_nearby_hospitals_vs_price_per_sqft.png")
    plt.close()
    
    # 14. Price by Furnished Status (boxplot)
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x='Furnished_Status', y='Price_in_Lakhs', palette='Set3')
    plt.title("Price by Furnished Status")
    plt.tight_layout()
    plt.savefig("eda_charts/14_price_by_furnished_status.png")
    plt.close()
    
    # 15. Price per sq ft by Facing direction (bar)
    plt.figure(figsize=(10,6))
    facing_price = df.groupby('Facing')['Price_per_SqFt'].mean()
    sns.barplot(x=facing_price.index, y=facing_price.values, palette='rocket')
    plt.title("Average Price per SqFt by Facing Direction")
    plt.tight_layout()
    plt.savefig("eda_charts/15_price_per_sqft_by_facing.png")
    plt.close()
    
    # 16. Owner Type distribution (pie or bar)
    plt.figure(figsize=(8,8))
    owner_counts = df['Owner_Type'].value_counts()
    plt.pie(owner_counts, labels=owner_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title("Owner Type Distribution")
    plt.tight_layout()
    plt.savefig("eda_charts/16_owner_type_distribution.png")
    plt.close()
    
    # 17. Availability Status distribution (pie or bar)
    plt.figure(figsize=(8,8))
    avail_counts = df['Availability_Status'].value_counts()
    plt.pie(avail_counts, labels=avail_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('deep'))
    plt.title("Availability Status Distribution")
    plt.tight_layout()
    plt.savefig("eda_charts/17_availability_status_distribution.png")
    plt.close()
    
    # 18. Parking Space vs Price (bar)
    plt.figure(figsize=(8,6))
    sns.barplot(data=df, x='Parking_Space', y='Price_in_Lakhs', palette='husl')
    plt.title("Parking Space vs Property Price")
    plt.tight_layout()
    plt.savefig("eda_charts/18_parking_space_vs_price.png")
    plt.close()
    
    # 19. Amenities vs Price_per_SqFt (bar)
    # Get primary amenity from the comma list
    df['Primary_Amenity'] = df['Amenities'].apply(lambda x: str(x).split(',')[0].strip())
    plt.figure(figsize=(12,6))
    sns.barplot(data=df, x='Primary_Amenity', y='Price_per_SqFt', palette='mako')
    plt.title("Primary Amenity vs Price per SqFt")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("eda_charts/19_amenities_vs_price_per_sqft.png")
    plt.close()
    
    # 20. Public Transport Accessibility vs Price_per_SqFt (bar)
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='Public_Transport_Accessibility', y='Price_per_SqFt', palette='crest')
    plt.title("Public Transport Accessibility vs Price per SqFt")
    plt.tight_layout()
    plt.savefig("eda_charts/20_public_transport_vs_price_per_sqft.png")
    plt.close()

    print("Successfully generated all 20 charts inside eda_charts/")

if __name__ == "__main__":
    main()

"""
TASK 2: Exploratory Data Analysis (EDA)
This script performs an EDA on the provided restaurant dataset to meet the task requirements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def main():
    # Load dataset
    try:
        df = pd.read_csv('Dataset .csv')
    except Exception as e:
        print(f"Couldn't load file: {e}")
        return

    print("Data shape:", df.shape)
    
    # Quick look at missing data
    missing = df.isnull().sum()
    print("\nMissing values:")
    print(missing[missing > 0])
    
    # --------------------------
    # Visualizations
    # --------------------------
    sns.set_theme(style="whitegrid")
    
    # Ratings vs Delivery and Booking
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.barplot(data=df, x='Has Online delivery', y='Aggregate rating', ax=ax1)
    ax1.set_title('Avg Rating: Online Delivery')
    
    sns.barplot(data=df, x='Has Table booking', y='Aggregate rating', ax=ax2)
    ax2.set_title('Avg Rating: Table Booking')
    
    plt.tight_layout()
    plt.show()

    # Look for weird ratings by price range
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Price range', y='Aggregate rating')
    plt.title('Rating Spread by Price Range')
    plt.show()

    # Basic correlation check
    plt.figure(figsize=(8, 6))
    num_cols = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(num_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlations')
    plt.show()

    # --------------------------
    # T-test for table bookings
    # --------------------------
    # Ignore the 0 ratings since those usually mean "not rated"
    df_rated = df[df['Aggregate rating'] > 0]
    
    booking_yes = df_rated[df_rated['Has Table booking'] == 'Yes']['Aggregate rating'].dropna()
    booking_no = df_rated[df_rated['Has Table booking'] == 'No']['Aggregate rating'].dropna()

    t_stat, p_val = stats.ttest_ind(booking_yes, booking_no, equal_var=False)
    
    print(f"\nT-test results (Table booking vs Rating):")
    print(f"t-stat: {t_stat:.3f}, p-val: {p_val:.4f}")
    if p_val < 0.05:
        print("Looks like table booking makes a difference.")
    else:
        print("No significant difference.")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import os  # We need this to interact with the operating system folders

def generate_retail_data():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
    products = ['Product_A', 'Product_B', 'Product_C']
    
    data = []
    for prod in products:
        base_demand = np.random.randint(50, 100)
        for date in dates:
            # Simulate seasonality (higher sales on weekends)
            weekday_effect = 1.3 if date.dayofweek > 4 else 1.0
            demand = int(base_demand * weekday_effect + np.random.normal(0, 10))
            data.append([date, prod, max(0, demand)])
            
    df = pd.DataFrame(data, columns=['Date', 'Product_ID', 'Sales'])
    
    # --- THE FIX ---
    # This line safely creates the 'data' folder if it doesn't exist yet
    os.makedirs('data', exist_ok=True)
    
    # Now it is safe to save the file
    df.to_csv('data/raw_sales.csv', index=False)
    print("✅ Dataset Created Successfully: data/raw_sales.csv")

if __name__ == "__main__":
    generate_retail_data()
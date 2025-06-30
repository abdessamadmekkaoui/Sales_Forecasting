import pandas as pd

# Load the CSV (replace 'input.csv' with your actual file)
df = pd.read_csv('merged.csv')

# 1. Remove completely empty rows and columns
df.dropna(how='all', inplace=True)
df.dropna(axis=1, how='all', inplace=True)

# 2. Drop the unwanted columns
columns_to_drop = [
    'Avg Basket Value','Avg Basket Value Excl Vat','Avg Basket Qty','Avg Item Value Incl',
    'Avg Item Value Excl', 'Sales Tax','Sales Incl'
]
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# 3. Extract date from 'Date_Time' column and create a new 'Date' column
df['Date1'] = pd.to_datetime(df['Date'], errors='coerce').dt.date

# 4. Drop the original 'Date_Time' column
df.drop(columns=['Date'], inplace=True)

# Optional: Drop rows where 'Date' couldn't be parsed
df.dropna(subset=['Date1'], inplace=True)

# Save the cleaned data
df.to_csv('cleaned.csv', index=False)

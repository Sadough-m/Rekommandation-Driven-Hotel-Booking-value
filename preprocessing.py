import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress warnings for cleaner execution
warnings.filterwarnings('ignore')

# ==========================================
# Configuration & Data Loading
# ==========================================
DATA_PATH = Path('hotel_reservations_pricing_revenue.csv')
RANDOM_STATE = 101

print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print('df', df.info());

# ==========================================
# Data Cleaning
# ==========================================
print("Cleaning data...")
df.drop_duplicates(inplace=True)
if 'id' in df.columns:
    df.drop_duplicates(subset=['id'], keep='first', inplace=True)

# Filter for confirmed, payment verified, fulfilled, and draft state
df = df[
    (df['is_confirmed'] == 1) & 
    (df['is_payment_verified'] == 1) & 
    (df['is_fulfilled'] == 1) & 
    (df['state'] == 'draft')
].copy()

# Keep only desktop users
df = df[df['os'] == 'desktop'].copy()

# Cap negative extra beds at 0
df['sum_room_extra_bed'] = df['sum_room_extra_bed'].clip(lower=0)

# Remove negative discounts and keep only successful, positive-revenue bookings
df = df[df['discount'] >= 0]
df = df[df['net_commission_revenue_eur'] >= 0]

# Handle missing values
worst_rank = df['hotel_rank'].max()
df['hotel_rank'] = df['hotel_rank'].fillna(worst_rank + 900)
df['hotel_stars'] = df['hotel_stars'].fillna(0)
df['sum_room_children'] = df['sum_room_children'].fillna(0)
df['sum_room_extra_bed'] = df['sum_room_extra_bed'].fillna(0)


# ==========================================
# Feature Engineering
# ==========================================
print("Engineering features...")

# Extract the Euro exchange rate and convert
df['Euro'] = df['currency_book'].apply(lambda x: json.loads(x).get("euro") if pd.notna(x) else None)
df['Euro'] = pd.to_numeric(df['Euro'], errors='coerce')

foreign_price_columns = [
    'price', 'discount', 'final_price', 'buy_price', 
    'original_sell_price', 'booking_price'
]

for col in foreign_price_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col] / df['Euro']

df[foreign_price_columns] = df[foreign_price_columns].round(2)

# Create boolean flags and date features
df['is_b2b'] = df['reservation_type'].apply(lambda x: 1 if x in ['b2b-normal', 'b2b-credit'] else 0)

for c in ['registered_date', 'booking_date', 'date_from', 'date_to']:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors='coerce')

df['lead_time_days'] = (df['date_from'] - df['booking_date']).dt.days
df['stay_nights'] = (df['date_to'] - df['date_from']).dt.days
df['booking_month'] = df['booking_date'].dt.month
df['booking_dow'] = df['booking_date'].dt.dayofweek

# Replace booking_date with registered_date for rows with negative lead_time_days
mask = df['lead_time_days'] < 0
df.loc[mask, 'booking_date'] = df.loc[mask, 'registered_date'].dt.strftime('%Y-%m-%d')

# keep hotel_accommodation_type for Gower distance method in unsupervised learning
df['hotel_accommodation'] = df['hotel_accommodation_type'].copy()
# One-hot encode accommodation type
df = pd.get_dummies(df, columns=['hotel_accommodation_type'], drop_first=False)
print('df', df.info());

# Define target variable
q75 = df['net_commission_revenue_eur'].quantile(0.75)
df['high_value_booking'] = (df['net_commission_revenue_eur'] >= q75).astype(int)

# ==========================================
# Final Formatting & Splitting
# ==========================================
print("Finalizing datasets...")

cols_to_drop = [
    'is_confirmed', 'is_payment_verified', 'is_fulfilled', 'state', 
    'cancellation_state', 'os', 'adult_count', 'child_count', 'infant_count', 
    'travel_type', 'utm_campaign', 'payment_currency_unit', 'reservation_type', 
    'currency_book', 'Euro', 'id', 'original_sell_price', 'price', 'buy_price', 
    'booking_price', 'registered_date',
    'hotel_accommodation_type_Hotel'
]

# Drop redundant/leakage columns safely
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
df.reset_index(drop=True, inplace=True)

leakage_columns = [
    'net_commission_revenue_eur', 
    'cost_of_rooms_eur', 
    'gross_booking_revenue_eur'
]

# Isolate Target (y) and Features (X)
y = df['high_value_booking']
X = df.drop(columns=leakage_columns + ['high_value_booking'])

# Convert integer IDs to categorical
for col in ['user_id', 'hotel_id', 'city_id']:
    if col in X.columns:
        X[col] = X[col].astype('category')

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print("Preprocessing complete.")
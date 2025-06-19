import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore") # Suppress warnings

# --- 1. Load the Data ---
try:
    # Assuming 'cleaned_dataset.csv' is in the same directory as the script
    df = pd.read_csv('cleaned_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'cleaned_dataset.csv' not found. Please ensure the file is in the same directory.")
    exit()

# --- 2. Data Preprocessing and Feature Engineering ---

# Convert 'date' column to datetime and set as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True) # Ensure chronological order

# Select relevant columns for forecasting.
data = df[['irradiance', 'temperature', 'humidity', 'wind_speed']]

# Handle missing values (if any)
if data.isnull().sum().sum() > 0:
    print(f"Warning: Missing values found. Imputing with forward/backward fill.")
    data = data.ffill().bfill()

# --- TEMPORARY: Reduce Data Size for Faster Testing ---

print(f"Original dataset size: {len(df)} samples")

data = data.tail(10000) # Using last 10,000 data points for quicker iteration
print(f"Reduced dataset size for testing: {len(data)} samples")


# --- Feature Engineering: Create Lagged Features ---
n_lags = 24 # Number of past time steps to use as features

X = [] # Features
y = [] # Target

for i in range(n_lags, len(data)):
    X.append(data['irradiance'].iloc[i-n_lags:i].values)
    y.append(data['irradiance'].iloc[i])

X = np.array(X)
y = np.array(y)

print(f"Shape of features (X): {X.shape}")
print(f"Shape of target (y): {y.shape}")

# --- 3. Split Data into Training and Testing Sets ---
train_size = int(len(X) * 0.8)
X_train, X_test = X[0:train_size], X[train_size:]
y_train, y_test = y[0:train_size], y[train_size:]

print(f"\nTraining data points (X_train, y_train): {len(X_train)}")
print(f"Test data points (X_test, y_test): {len(X_test)}")

# --- 4. Scale Features ---
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_reshaped = y_train.reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)

y_train_scaled = scaler_y.fit_transform(y_train_reshaped).flatten()
y_test_scaled = scaler_y.transform(y_test_reshaped).flatten()


# --- 5. Implement and Fit the SVR Model ---
print("\nFitting SVR model...")

svr_model = SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale') 
svr_model.fit(X_train_scaled, y_train_scaled)
print("SVR model fitted successfully.")

# --- 6. Make Predictions ---
predictions_scaled = svr_model.predict(X_test_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

test_index_start = n_lags + (len(df) - len(data)) + train_size # Adjusted for subsetting
test_index = df.index[test_index_start : test_index_start + len(y_test)]


# --- 7. Evaluate the Model ---
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-8))) * 100

print(f"\nModel Performance Metrics:")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.3f}%")

# --- 8. Visualize Results ---
plt.figure(figsize=(15, 7))

# Plot actual training data

actual_train_index_start = n_lags + (len(df) - len(data))
actual_train_index = df.index[actual_train_index_start : actual_train_index_start + train_size]
plt.plot(actual_train_index, y_train, label='Training Data (Actual)', color='blue', alpha=0.7)


# Plot actual test data


# Plot SVR predictions


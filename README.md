# Solar Irradiance Forecasting with SVR

This project implements a Support Vector Regression (SVR) model to forecast solar irradiance based on historical meteorological data.

## Project Structure

* `cleaned_dataset.csv`: Your cleaned historical meteorological dataset, expected to be in the same directory as the Python script. It should contain at least 'date' and 'irradiance' columns, and optionally 'temperature', 'humidity', and 'wind_speed'.
* `requirements.txt`: Lists all necessary Python dependencies.
* `.gitignore`: Configured to ignore the virtual environment folder.

## Setup and Installation

Follow these steps to set up the project environment and run the model:

1. **Fork the REpo**
2.  **Clone the Repository :**
    If you're using Git, clone this repository to your local machine. Otherwise, ensure all project files are in the same directory.

3.  **Create a Virtual Environment**
    
4.  **Activate the Virtual Environment:**

    

5.  **Install Dependencies:**
    With your virtual environment activated, install all required libraries using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```


## How to Run the Project

1.  **Activate your virtual environment** (if not already active, see step 3 above).
2.  **Navigate to the project directory** in your terminal.
3.  **Run the Python script**

    

The script will load the data, preprocess it, train the SVR model, make predictions, evaluate its performance, and display a plot of the actual vs. predicted solar irradiance values.

## Model Details

* **Model**: Support Vector Regression (SVR)
* **Target Variable**: 'irradiance'
* **Features**: Lagged values of 'irradiance' (default `n_lags = 24`, representing the last 24 readings/hours).
* **Hyperparameters (Current)**: `kernel='rbf'`, `C=10`, `epsilon=0.1`, `gamma='scale'`
    *(These parameters were adjusted for faster training on a subset of the data. For full dataset performance, hyperparameter tuning is recommended.)*
* **Evaluation Metrics**: RMSE, MAE, MAPE

---


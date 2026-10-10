# Wind Power Forecasting

A machine-learning project for forecasting wind-turbine power generation from meteorological forecasts and historical production data.

## What this project demonstrates

- Exploratory analysis and cleaning of time-series weather and power data
- Feature engineering for wind speed, wind direction, and cyclical time effects
- A scikit-learn preprocessing pipeline with scaling and one-hot encoding
- Model comparison and hyperparameter tuning using cross-validation
- Forecast generation and evaluation with the R² metric

The workflow is designed to turn raw weather inputs into reproducible power forecasts. Wind vectors are represented as speed and direction, while wind-speed bands and seasonal/time features help the models capture the turbine’s nonlinear power curve.

## Project structure

- [`Notebooks/`](./Notebooks/) — exploration, preprocessing, model selection, and forecast evaluation
- [`process_data.py`](./process_data.py) — reusable feature engineering and preprocessing helpers
- [`Data/`](./Data/) — training data, forecast inputs, templates, and evaluation targets

## Run locally

```bash
pip install -r requirements.txt
```

Run the notebooks in order:

1. `1_data_exploration_cleaning.ipynb`
2. `2_data_exploration.ipynb`
3. `3_feature_preprocessing_pipeline_and_model_selection.ipynb`
4. `4_forecast_evaluation.ipynb`

This project was developed as an applied forecasting exercise, combining data analysis, feature engineering, model evaluation, and reproducible experimentation.

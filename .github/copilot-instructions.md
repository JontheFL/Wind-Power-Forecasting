# Wind Power Forecasting - AI Coding Guidelines

## Project Overview
This is a machine learning project for wind power forecasting using historical wind data and power generation. The pipeline processes raw meteorological data through exploration, feature engineering, model training, and evaluation stages.

## Architecture & Data Flow
- **Data Sources**: `TrainData_A.csv` (training), `WeatherForecastInput_A.csv` (forecast input), `Solution_A.csv` (ground truth)
- **Processing Pipeline**: Raw data → Missing value imputation → Polar coordinate conversion → Feature engineering → Model training → Forecast generation
- **Output Format**: `ForecastTemplate_A.csv` with `TIMESTAMP` and `FORECAST` columns
- **Evaluation Metric**: R² score comparing forecasts against `Solution_A.csv`

## Key Data Transformations

### Polar Coordinate Conversion
Convert Cartesian wind vectors (U,V) to polar coordinates:
```python
wind_speed = np.sqrt(U**2 + V**2)
wind_direction = np.arctan2(V, U) * (180 / np.pi)
wind_direction = (wind_direction + 360) % 360  # Normalize to 0-360°
```
- 0° = East, 90° = North, 180° = West, 270° = South
- Use 100m height data (`wind_speed_100m`, `wind_direction_100m`) for primary features

### Missing Value Strategies
Three imputation methods for missing POWER values:
- **Persistence**: Forward/backward fill (`fillna(method='ffill').fillna(method='bfill')`)
- **Moving Average**: Rolling mean with window size (e.g., `rolling(window=5, min_periods=1).mean()`)
- **Hourly Climatology**: Mean power per hour-of-day across all data

### Feature Engineering Patterns

#### Wind Speed Binning
```python
speed_bins = [0, 4, 20, 50]
speed_labels = ['cut-in', 'ramp-up', 'cut-out']
wind_speed_bin = pd.cut(wind_speed_100m, bins=speed_bins, labels=speed_labels)
```
- **cut-in**: 0-4 m/s (power rises from 0)
- **ramp-up**: 4-20 m/s (power increases to max)
- **cut-out**: >20 m/s (power plateaus or decreases)

#### Wind Direction Binning
```python
dir_bins = [0, 90, 180, 270, 360]
dir_labels = ['NE', 'SE', 'SW', 'NW']
wind_dir_bin = pd.cut(wind_direction_100m, bins=dir_bins, labels=dir_labels)
```
- Critical for accuracy: Western directions (SW/NW) show reduced power output

#### Cyclical Time Features
```python
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
```
- Prevents discontinuity at year/hour boundaries
- Use both sin and cos for complete cyclical representation

## Development Workflow

### Data Processing
1. Load data with `pd.read_csv()` and convert TIMESTAMP: `pd.to_datetime(timestamp, format='%Y%m%d %H:%M')`
2. Handle missing values using one of three strategies above
3. Convert to polar coordinates and drop U/V columns
4. Save processed data to `Data/Processed/` directory

### Feature Pipeline
```python
# Preprocessing pipeline pattern
numerical_features = ['wind_speed_100m']
categorical_features = ['wind_speed_bin', 'wind_dir_bin']
passthrough_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(), categorical_features),
    ('pass', 'passthrough', passthrough_features)
])
```

### Model Development
- Use scikit-learn pipelines for preprocessing + model
- Train on processed data from `Data/Processed/`
- Generate forecasts in `ForecastTemplate_A.csv` format
- Evaluate using R² score against `Solution_A.csv`

## Recommended Models

### Primary Recommendation: Gradient Boosting (XGBoost/LightGBM)
```python
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

# XGBoost with tuned parameters for wind power forecasting
xgb_model = XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Cross-validation with time series split
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(xgb_model, X_train, y_train, cv=tscv, scoring='r2')
```

### Alternative: Random Forest
```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
```

### Baseline: Linear Regression
```python
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
```

## Model Comparison Strategy
```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

scoring = {
    'r2': make_scorer(r2_score),
    'mae': make_scorer(mean_absolute_error, greater_is_better=False)
}

results = {}
for name, model in models.items():
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    cv_results = cross_validate(pipeline, X_train, y_train, cv=5, scoring=scoring)
    results[name] = cv_results
```

## Key Insights from Exploration
- **Strong correlation**: Wind speed is primary predictor of power output
- **Directional effects**: Western winds produce less power than eastern winds
- **Seasonal patterns**: Higher power generation in winter months
- **No diurnal patterns**: Wind speed/power don't vary significantly by hour-of-day
- **S-curve relationship**: Power curve shows cut-in, ramp-up, and cut-out regions

## File Organization
- `Notebooks/`: Jupyter notebooks for exploration and development
- `Data/`: Raw data and ground truth files
- `Data/Processed/`: Cleaned and transformed datasets
- `src/`: (Currently empty) - intended for production Python modules

## Common Patterns
- Always convert timestamps to datetime64[ns] type
- Use polar coordinates for analysis and modeling
- Include wind direction features despite low linear correlation
- Apply cyclical encoding for temporal features
- Save intermediate results to `Data/Processed/` for reproducibility
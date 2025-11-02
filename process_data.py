def create_features(wind):
    """
    Create features to improve wind power forecasting and splits the data in features and target variable.
    """
    wind_feat = wind.copy()
    wind_feat['TIMESTAMP'] = pd.to_datetime(wind_feat['TIMESTAMP']) 
    
    # Look for POWER column if exists drop it and store it as target variable
    if 'POWER' in wind_feat.columns:
        y_train = wind['POWER']
        wind_feat = wind_feat.drop(columns=['POWER'])
    else:
        y_train = None

    # 1. Wind speed bins using 100m (linearizing S-curve)
    # Based on the plots, we define three regions:
    # a. Cut-in: 0-4 m/s
    # b. Ramp-up: 4-20 m/s
    # c. Cut-out: >20 m/s

    speed_bins = [0, 4, 20, 50]
    speed_labels = ['cut-in', 'ramp-up', 'cut-out']

    wind_feat['wind_speed_bin'] = pd.cut(wind_feat['wind_speed_100m'], bins=speed_bins, labels=speed_labels, right=False, include_lowest=True)

    # 2. Wind direction bins
    dir_bins = [0, 90, 180, 270, 360]
    dir_labels = ['NE', 'SE', 'SW', 'NW']
    wind_feat['wind_dir_bin'] = pd.cut(wind_feat['wind_direction_100m'], bins=dir_bins, labels=dir_labels, right=False, include_lowest=True)

    # 3. Cyclical Time Features (hour)
    wind_feat['hour'] = wind_feat['TIMESTAMP'].dt.hour
    wind_feat['hour_sin'] = np.sin(2 * np.pi * wind_feat['hour'] / 24)
    wind_feat['hour_cos'] = np.cos(2 * np.pi * wind_feat['hour'] / 24)

    # 4. Cyclical Time Features (month)
    wind_feat['month'] = wind_feat['TIMESTAMP'].dt.month
    wind_feat['month_sin'] = np.sin(2 * np.pi * wind_feat['month'] / 12)
    wind_feat['month_cos'] = np.cos(2 * np.pi * wind_feat['month'] / 12)

    return wind_feat , y_train


def get_preprocessor():
    """
    Define and return preprocessing pipeline.
    """
    # 1. Define feature lists
    numerical_features = ['wind_speed_100m']
    categorical_features = ['wind_speed_bin', 'wind_dir_bin']
    passthrough_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']

    # 2. Define the transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 3. Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('pass', 'passthrough', passthrough_features)
        ]
    )

    return preprocessor

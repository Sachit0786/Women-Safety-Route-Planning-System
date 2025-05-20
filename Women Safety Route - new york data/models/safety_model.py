import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
import matplotlib.pyplot as plt
import os
import joblib

def prepare_data(data):
    """
    Prepare data for model training by separating features and target.
    """
    # Drop non-feature columns
    features_to_drop = ['latitude', 'longitude']
    X = data.drop(columns=[col for col in features_to_drop if col in data.columns])
    
    print("Calculating 'safety score' using fallback weights...")
    data = data.copy()
    data['safety_score'] = (
            data.get('cameras', 0) * 0.90 +
            data.get('street_lights', 0) * 0.75 +
            data.get('public_transport', 0) * 0.65 +
            data.get('police_stations', 0) * 0.80 +
            data.get('crime_rate', 0) * -0.85 +
            data.get('accidents', 0) * -0.60 +
            data.get('shops', 0) * 0.55 +
            data.get('construction_zones', 0) * -0.40 +
            data.get('crowd_density', 0) * 0.50 +
            data.get('parks_recreation', 0) * 0.10 +
            data.get('population_density', 0) * 0.30 +
            data.get('traffic_density', 0) * 0.25 +
            data.get('market_areas', 0) * 0.40 +
            data.get('time_of_day', 0) * -0.35 +
            data.get('sidewalk_presence', 0) * 0.50 +
            data.get('is_festival', 0) * 0.20 +
            data.get('hospitality_venues', 0) * 0.10 +
            data.get('emergency_services', 0) * 0.85 +
            data.get('is_holiday', 0) * -0.15 +
            data.get('is_night', 0) * -0.75 +
            data.get('day_of_week', 0) * 0.15 +
            data.get('religious_places', 0) * 0.10 +
            data.get('educational_institutions', 0) * 0.20
    )

    # Normalize
    min_val = data['safety_score'].min()
    max_val = data['safety_score'].max()
    data['safety_score'] = (data['safety_score'] - min_val) / (max_val - min_val + 1e-6)

    y = data['safety_score']
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'safety_scaler.joblib')
    data.to_csv("data/data_with_safety_score.csv", index=False)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler



def build_advanced_model(input_dim):
    """
    Build an advanced neural network model with dropout and batch normalization.
    """
    model = Sequential([
        Dense(128, activation='sigmoid', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='sigmoid'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='sigmoid'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(16, activation='sigmoid'),
        BatchNormalization(),
        
        Dense(1, activation='sigmoid')  # Output between 0 and 1
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def model_details(data, epochs=500, batch_size=32):
    """
    Train an advanced neural network model for safety prediction.
    """
    # Prepare the data
    X_train, X_test, y_train, y_test, scaler = prepare_data(data)
    
    # Build the model
    model = build_advanced_model(X_train.shape[1])
    
    # Define callbacks with adjusted parameters
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=30,  # Increased from 20 to 30
        restore_best_weights=True,
        min_delta=1e-4  # Added minimum change threshold
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Changed from 0.2 to 0.5 for gentler reduction
        patience=15,  # Increased from 10 to 15
        min_lr=1e-6  # Adjusted minimum learning rate
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate the model
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Training Loss: {train_loss[0]:.4f}, Training MAE: {train_loss[1]:.4f}")
    print(f"Testing Loss: {test_loss[0]:.4f}, Testing MAE: {test_loss[1]:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save the model
    model.save('models/safety_model.h5')
    print("Model saved as 'models/safety_model.h5'")
    
    # Analyze feature importance
    feature_columns = list(scaler.feature_names_in_)

    analyze_feature_importance(model, data, feature_columns)
    return

def plot_training_history(history):
    """
    Plot the training history.
    """
    # Create directory for plots if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Plot training & validation loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/training_history.png')
    plt.close()

def analyze_feature_importance(model, data, feature_columns):
    """
    Analyze the importance of each feature in the model with focus on women safety factors.
    """
    # Define safety weights for different features
    safety_weights = {
        "cameras": 0.90,  # High deterrent for crime; enhances monitoring and safety.
        "street_lights": 0.75,  # Well-lit areas improve visibility and reduce risk.
        "police_stations": 0.80, # Areas with more stations are more safe
        "public_transport": 0.65,  # Indicates presence of people and easier escape options.
        "crime_rate": -0.85,  # Direct indicator of unsafe areas; higher crime = less safe.
        "accidents": -0.60,  # Reflects dangerous or poorly maintained areas.
        "shops": 0.55,  # Commercial activity implies people, lighting, and social safety.
        "construction_zones": -0.40,  # Often isolated, poorly lit, and unsafe.
        "crowd_density": -0.60,  # Higher crowd density increases chances of theft or stampede
        "parks_recreation": 0.10,  # Can be safe in the day but unsafe at night; neutral overall.
        "population_density": 0.30,  # Denser areas have more people and visibility.
        "traffic_density": 0.25,  # Some traffic means activity; too much can be chaotic.
        "distance": -0.20,  # Longer routes may pass through more isolated or risky areas.
        "market_areas": 0.40,  # Generally busy and well-lit during open hours.
        "time_of_day": -0.35,  # Safety decreases significantly during late hours.
        "sidewalk_presence": 0.50,  # Safe pedestrian infrastructure reduces road risks.
        "is_festival": 0.20,  # Increases public presence and surveillance, though crowd control varies.
        "hospitality_venues": 0.10,  # Can offer lighting and help but may be risky late at night.
        "emergency_services": 0.85,  # Proximity to help in emergencies greatly increases safety.
        "is_holiday": -0.25,  # Empty streets and closed shops may reduce safety.
        "is_night": -0.85,  # Nighttime poses a major safety risk due to lower visibility and activity.
        "day_of_week": 0.15,  # Minor influence; mostly neutral compared to other contextual factors.
        "religious_places": 0.10,  # Often seen as peaceful/safe areas, though variable.
        "educational_institutions": 0.20  # Safe during working hours; inactive at night but generally secure.
    }
    
    # Use the same feature columns that were used during training
    feature_names = feature_columns
    
    # Load the scaler
    try:
        scaler = joblib.load('safety_scaler.joblib')
    except:
        print("Warning: Scaler not found. Using MinMaxScaler.")
        scaler = MinMaxScaler()
        # Fit the scaler with the actual data
        X = data[feature_columns]
        scaler.fit(X)
    
    # Get the mean values of each feature from the actual data
    mean_values = data[feature_columns].mean().values
    
    # Calculate feature importance using mean values as baseline
    importance = []
    for i in range(len(feature_names)):
        feature_name = feature_names[i]
        
        # Create a copy of the mean values
        input_copy = mean_values.copy()
        
        # Get the actual min and max values for this feature
        feature_min = data[feature_columns].iloc[:, i].min()
        feature_max = data[feature_columns].iloc[:, i].max()
        
        # Calculate importance using multiple points between min and max
        num_points = 10
        feature_values = np.linspace(feature_min, feature_max, num_points)
        predictions = []
        
        for value in feature_values:
            # Set the feature to the current value
            input_copy[i] = value
            
            # Reshape and scale
            input_df = pd.DataFrame([input_copy], columns=scaler.feature_names_in_)
            input_scaled = scaler.transform(input_df)

            # Get prediction
            pred = model.predict(input_scaled, verbose=0)[0][0]
            predictions.append(pred)
        
        # Calculate importance as the range of predictions
        pred_range = max(predictions) - min(predictions)
        
        # Apply safety weight if feature has one
        weight = safety_weights.get(feature_name, 1.0)
        weighted_importance = pred_range * weight
        
        importance.append((feature_name, weighted_importance))
    
    # Sort by importance
    importance.sort(key=lambda x: x[1], reverse=True)
    
    # Plot the top 15 features
    plt.figure(figsize=(12, 8))
    features = [x[0] for x in importance[::]]
    values = [x[1] for x in importance[::]]
    
    # Create a color map based on safety weights
    colors = ['#2ecc71' if safety_weights.get(f, 1.0) >= 1.3 else 
              '#f1c40f' if safety_weights.get(f, 1.0) >= 1.2 else 
              '#e74c3c' for f in features]
    
    plt.barh(features, values, color=colors)
    plt.title('Importance of Safety Factors (Women Safety Focus)')
    plt.xlabel('Weighted Importance Score')
    plt.tight_layout()
    plt.savefig('output/feature_importance.png')
    plt.close()
    
    # Print the top 10 features with their weights
    print("\nImportant Safety Factors (Women Safety Focus):")
    for i, (feature, value) in enumerate(importance[::]):
        weight = safety_weights.get(feature, 1.0)
        print(f"{i+1}. {feature}: {value:.4f} (Weight: {weight:.1f}x)")
    
    return importance 
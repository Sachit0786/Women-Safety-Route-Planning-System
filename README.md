# 🛡️ Women Safety Route Planning System

A machine learning-powered geospatial application that suggests the safest path between two locations based on crime data and other safety indicators.

## 🚀 Features

- **Deep Learning Model**  
  Built with TensorFlow/Keras using a multi-layer neural network (128-64-32-16-1) with batch normalization and dropout layers for better generalization.

- **Smart Routing**  
  Implements Dijkstra’s algorithm with dual optimization (safety & distance) using NetworkX and OSMnx.

- **Interactive Maps**  
  Visualizes routes and safety heatmaps using Folium and OpenStreetMap APIs.

- **Safety Scoring System**  
  Combines multiple safety metrics using Pandas and NumPy for weighted scoring.

- **Robust Design**  
  Includes custom evaluation tools, HTML/CSS-based UI components, and fallback mechanisms for unreliable data sources.

## 📌 Technologies Used

- Python, TensorFlow, Keras, NumPy, Pandas
- NetworkX, OSMnx, Folium, OpenStreetMap APIs
- Scikit-learn, HTML/CSS

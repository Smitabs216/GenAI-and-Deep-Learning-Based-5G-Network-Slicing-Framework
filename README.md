# GenAI and Deep Learning Based 5G Network Slicing Framework

## Overview

This project presents an AI-driven 5G Network Slicing Framework that combines Deep Learning, Deep Reinforcement Learning (DRL), and Generative AI to optimize network slicing decisions in 5G environments.

The framework predicts network traffic demand using LSTM, performs intelligent resource allocation using DRL and Convex Optimization, applies admission control based on QoS requirements, and recommends the most suitable network slice for real-time applications.

---

## Key Features

- LSTM-based Traffic Prediction
- Deep Reinforcement Learning (DQN) Resource Allocation
- Convex Optimization Resource Allocation
- QoS-aware Admission Control
- AI-Based Slice Selection
- Generative AI Recommendation Engine
- Live Network Analysis Demo
- Network Comparison Module
- Performance Visualization and Analytics

---

## Network Slices

### eMBB (Enhanced Mobile Broadband)

Suitable for:

- Netflix
- YouTube
- HD Video Streaming
- AR/VR Applications
- High-Speed Internet Access

Requirements:

- High Throughput
- Moderate Latency

---

### URLLC (Ultra-Reliable Low-Latency Communication)

Suitable for:

- Cloud Gaming
- Video Conferencing
- Remote Surgery
- Industrial Automation
- Autonomous Vehicles

Requirements:

- Ultra-Low Latency
- Extremely Low Packet Loss

---

### mMTC (Massive Machine Type Communication)

Suitable for:

- IoT Devices
- Smart Agriculture
- Smart Cities
- Sensor Networks
- Machine-to-Machine Communication

Requirements:

- Massive Connectivity
- Low Data Rate

---

## Project Workflow

1. Load and preprocess network dataset
2. Train LSTM models for traffic prediction
3. Predict throughput, delay, and packet loss
4. Perform resource allocation using:
   - Greedy Allocation
   - Convex Optimization
   - Deep Reinforcement Learning (DQN)
5. Apply QoS-based admission control
6. Select optimal network slice
7. Generate AI recommendations
8. Visualize performance metrics and results

---

## Live Demo Module

The project includes a live demonstration module where users can manually provide:

- Throughput (Mbps)
- Delay (ms)
- Packet Loss Rate (PLR)

The AI engine automatically:

- Identifies the most suitable network slice
- Displays application recommendations
- Performs network comparison
- Generates visual analysis graphs

Example:

Input:

Throughput = 32 Mbps  
Delay = 31 ms  
PLR = 0.00001

Output:

Selected Slice = eMBB

Recommended Applications:

- Netflix
- YouTube
- AR/VR

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Deep Reinforcement Learning (DQN)
- Generative AI

---

## Performance Evaluation

The framework evaluates:

- MAE
- RMSE
- R² Score
- Resource Utilization
- Throughput
- Delay
- Packet Loss Rate
- Admission Control Performance

---

## Generated Outputs

- LSTM Training & Validation Curves
- Prediction vs Actual Analysis
- Resource Allocation Comparison
- DRL Reward Analysis
- Admission Control Results
- Slice Selection Visualization
- Throughput Comparison Charts
- Excel Summary Report

---

## Repository Structure

```text
main.py
lstm_model.py
resource_allocation.py
admission_control.py
slice_selector.py
live_demo.py
live_compare.py
genai_recommendation.py
results.py

dataset.csv
Results_Summary.xlsx
```

## Author

Soumy

GenAI and Deep Learning Based 5G Network Slicing Framework

# Parking-Occupancy-Model
## Introduction
Curbside parking occupancy prediction is an important aspect of parking management for cities. However, training models to predict every street in a city would require collecting a lots of ground truth data. The collection of such data would cost a lots of time and money. This page explore the idea of finding blocks that are more meaningful for the model to learn from, therefore can potentially reduce the amount data require to be collected. 

## Background
The problem of finding the most meaningful blocks to train a model can treated as a type of multi-armed bandit problem. A collection of blocks can be cluster into groups with similar occupancy characteristics and each cluster can be considered as a machine. Each machine would give the model one data point from the corresponding blocks cluster to train. The goal of the model agent is to maximize the prediction accuracy on a separate testing data. One solution is to represent the entire block clusters system as a Markov Decision Process, where .In the end after training, the model agent would have 

## Implementation
The base predictive model used is a ridge regression with a radial basis function kernel. This particular model is chosen because it is simple to train with little data points, while still having decent accuracy. 

# Parking-Occupancy-Model
## Introduction
Curbside parking occupancy prediction is an important aspect of parking management for cities. However, training models to predict every street in a city would require collecting a lots of ground truth data. The collection of such data would cost a lots of time and money. This page explore the idea of finding blocks that are more mearningful for the model to learn from, therefore can potentially reduce the amount data require to be collected. 

## Background
The base predictive model used is a ridge regression with a radial basis function kernel. This particular model is chosen because it is simple to train with little data points, while still having decent accuracy. 

# Parking-Occupancy-Model
## Introduction
Curbside parking occupancy prediction is an important aspect of parking management for cities. However, training models to predict every street in a city would require collecting a lots of ground truth data. The collection of such data would cost a lots of time and money. This page explore the idea of finding blocks that are more meaningful for the model to learn from, therefore can potentially reduce the amount data require to be collected. 

## Background
The problem of finding the most meaningful blocks to train a model can treated as a type of multi-armed bandit problem. A collection of blocks can be cluster into groups with similar occupancy characteristics and each cluster can be considered as a machine. Each machine would give the model one data point from the corresponding blocks cluster to train. The goal of the model agent is to maximize the prediction accuracy on a separate testing data. One solution is to represent the entire block clusters system as a Markov Decision Process, where each state is a combination of blocks from all the different clusters. The action is therefore adding and removing block from each of the clusters. The optimal solution can be found using Q-learning algorithm. After training, the resulting combination of blocks state will tell us which clusters is more impactful in improving the model accuracy.

## Implementation
The parking occupancy data used is gathered and processed from Seattle Department of Transportation (SDOT). The naive and ground truth data consist of 12 different blocks around Seattle, where each block has data collected for 7 day (03/21/22 - 03/26/22 + 03/28/22) from 8am to 8pm with a time interval of 1 minute. In total there are 84 whole day of parking occupancy data from 8am to 8pm across the 12 blocks and 7 days for each block. The data is split into a training and testing set where there are 72 whole day of training and 12 whole day of testing data.

The base predictive model used is a ridge regression with a radial basis function kernel. This particular model is chosen because it is simple to train with little data points, while still having decent accuracy. The sklearn kernel_ridge package is used to implement the model with lambda value of 0.01 and gamma value of 0.001.  The accuracy is computed using the mean absolute error:

$$ Accuracy = 1 - \frac{ \sum\limits_{i=0}^n \vert y_{i} - x_{i} \vert }{n} $$


The K-mean clustering is used to group the blocks into 6 individual bins. The number of bin is chosen arbitary and the number 6 is used because it is the number of different sub-area the 12 blocks reside in, University District, Belltown, Uptown Triangle, Denny Triangle, Pike-Pine and South Lake Union. There are no definitive correlation between the parking occupancy and sub-area of the blocks, however the assumption that there is a correlation is used to derive the number of bin. The naive training parking occupancy are clustered using K-mean into 6 different bins, one of the resulting clusters when mapped on to the first 2 principal components is shown in figure 1. Then every training data point get assigned to there corresponding bin based on the resulting K-mean clustering.

<p align="center">
  <img src="https://github.com/hhuynh000/Parking-Occupancy-Model/blob/main/figures/K-mean.png" width="600"/>
</p>
<p align="center">
  Figure 1. K-mean Clustering on Naive Parking Occupancy
</p>

After grouping the blocks into bin, the bins are setup as state in the Markov Decision Process using the class BanditMDP. The initial state of the prediction model in the Markov Decision Process is having zero block ([0,0,0,0,0,0]). Also, The model requires a goal state to be set, which indicate the total number of blocks from all bins used to train the model. A smaller goal state will take in account less of the overall data but the solution is faster to find, whereas a larger goal state will take into account most of the data but the solution will be slower to find. When the model reaches a goal state in the Markov Decision Process, the model can "exit" and claim a reward which is equal to the model accuracy when tested on the testing data. After the model reaches the goal state the model will start over with zero block. Every step the model take a data point from one of the bins can be added or removed from the data pool being used to train the model. The reward given at any non "exit" action is -0.01 in order to discourage the model from repeating state it already been to. 

The Q-learning algorithm is used train the model into finding the optimal combination of bins which will produce the highest accuracy. The algorithm is an iterative process to find Q-value as the model explore the state space. The new sample estimate of Q-value given a sample of the current state s, the action a, the next state s' and the reward r (s,a,s',r) is described in Equation 1. The new updated Q-value incorporate the new estimate into a running average describes by Equation 2, where alpha is the percentage of the new estimate to update the Q-value. The default $\alpha$ value used is 0.9.

$$ sample = R(s,a,s') + \gamma \max\limits_{a'} Q(s',a') \quad \quad [1] $$


$$ Q(s,a) = (1-\alpha)Q(s,a) + \alpha (sample) \quad \quad [2] $$

In order to encourage exploration, an epsilon greedy function is used to incorporate a possibility of a random action taken by the model. There is a probablity of $\epsilon$ to perform a random action based on the current state and there is a probablity of $1-\epsilon$ of acting based on the current policy. The default $\epsilon$ value used is 0.5.

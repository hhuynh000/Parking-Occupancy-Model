# Parking-Occupancy-Model
## Introduction
Curbside parking occupancy prediction is an important aspect of parking management for cities. However, training models to predict every street in a city would require collecting a lots of ground truth data. The collection of such data would cost a lots of time and money. This page explore the idea of finding blocks that are more meaningful for the model to learn from, therefore can potentially reduce the amount data require to be collected. 

## Background
The problem of finding the most meaningful blocks to train a model can treated as a type of multi-armed bandit problem. A collection of blocks can be cluster into groups with similar occupancy characteristics and each cluster can be considered as a machine. Each machine would give the model one data point from the corresponding blocks cluster to train. The goal of the model agent is to maximize the prediction accuracy on a separate testing data. One solution is to represent the entire block clusters system as a Markov Decision Process, where each state is a combination of blocks from all the different clusters. The action is therefore adding and removing block from each of the clusters. The optimal solution can be found using Q-learning algorithm. After training, the resulting combination of blocks state will tell us which clusters is more impactful in improving the model accuracy.

## Implementation
The parking occupancy data used is gathered and processed from Seattle Department of Transportation (SDOT). The naive and ground truth data consist of 12 different blocks around Seattle, where each block has data collected for 7 day (03/21/22 - 03/26/22 + 03/28/22) from 8am to 8pm with a time interval of 1 minute. In total there are 84 whole day of parking occupancy data from 8am to 8pm across the 12 blocks and 7 days for each block. The data is split into a training and testing set where there are 72 whole day of training and 12 whole day of testing data. The occupancy data is normalized as an occupancy percentage based on the max parking capacity of each individual blocks.

The base predictive model used is a ridge regression with a radial basis function kernel. This particular model is chosen because it is simple to train with little data points, while still having decent accuracy. The sklearn kernel_ridge package is used to implement the model with lambda value of 0.01 and gamma value of 0.001.  The accuracy is computed using the mean absolute error:

$$ Accuracy = 1 - \frac{ \sum\limits_{i=0}^n \vert y_{i} - x_{i} \vert }{n} $$


The K-mean clustering is used to group the blocks into 6 individual bins. The number of bin is chosen arbitary and the number 6 is used because it is the number of different sub-area the 12 blocks reside in, University District, Belltown, Uptown Triangle, Denny Triangle, Pike-Pine and South Lake Union. There are no definitive correlation between the parking occupancy and sub-area of the blocks, however the assumption that there is a correlation is used to derive the number of bin. The naive training parking occupancy are clustered using K-mean into 6 different bins, one of the resulting clusters when mapped on to the first 2 principal components is shown in figure 1. Then every training data point get assigned to there corresponding bin based on the resulting K-mean clustering.

<p align="center">
  <img src="https://github.com/hhuynh000/Parking-Occupancy-Model/blob/main/figures/K-mean.png" width="600"/>
</p>
<p align="center">
  Figure 1. K-mean Clustering on Naive Parking Occupancy
</p>

After grouping the blocks into bin, the bins are setup as state in the Markov Decision Process using the class BanditMDP. The initial state of the prediction model in the Markov Decision Process is having zero block ([0,0,0,0,0,0]). Also, The model requires a goal state to be set, which indicate the total number of blocks from all bins used to train the model. A smaller goal state will take in account less of the overall data but the solution is faster to find, whereas a larger goal state will take into account most of the data but the solution will be slower to find. When the model reaches a goal state in the Markov Decision Process, the model can "exit" and claim a reward which is equal to the model accuracy when tested on the testing data. After the model reaches the goal state the model will start over with zero block. At every step the model can add or remove a data point from the data pool being used to train the model. The reward given at any non "exit" action is -0.01 in order to discourage the model from repeating state it already been to. 

The Q-learning algorithm is used train the model into finding the optimal combination of bins which will produce the highest accuracy. The algorithm is an iterative process to find Q-value as the model explore the state space. The new sample estimate of Q-value given a sample of the current state s, the action a, the next state s' and the reward r (s,a,s',r) is described in Equation 1. The new updated Q-value incorporate the new estimate into a running average describes by Equation 2, where alpha is the percentage of the new estimate to update the Q-value. The default $\alpha$ value used is 0.9.

$$ sample = R(s,a,s') + \gamma \max\limits_{a'} Q(s',a') \quad \quad [1] $$


$$ Q(s,a) = (1-\alpha)Q(s,a) + \alpha (sample) \quad \quad [2] $$

In order to encourage exploration, an epsilon greedy function is used to incorporate a possibility of a random action taken by the model. There is a probablity of $\epsilon$ to perform a random action based on the current state and there is a probablity of $1-\epsilon$ of acting based on the current policy. The default $\epsilon$ value used is 0.5 to encourage exploration because the only big reward is at the goal states.

## Result
The Q-learning algorithm is ran for 10,000,000 iterations with a goal state of 24 samples which take about 2 minutes and 30 seconds to run. Then the 3 top bins in term of number of sample from the solution are used to train the occupancy model and test on the testing data. Likewise, the bottom 3 bins are used to train the occupancy model and test on the testing data. The number of top bins to chose from is arbitary chosen and 3 is a middle cut off in the case with 6 total bins. In addition, the accuracy of the occupancy prediction when the model trained on all the training data and the accuracy of the naive occupancy are computed for reference. The amount of avaliable data is limited, therefore testing different split of the data is necessary to see if this impact the solution. The results from varying the random_state of the train_test_split function from sklearn, while keeping K-mean random state the same is shown in Table I.

| Split Random State | Full Training Accuracy | Naive Occupancy Accuracy | Top 3 Bins Accuracy | Bottom 3 Bins Accuracy | Bin Distribution | 
| --- | --- | --- | --- | --- | --- |
| 10 | 87.71% | 59.96% | 87.35% [1, 5, 2] | 73.36% [0, 3, 4] | [20, 14, 15, 2, 10, 11] |
| 50 | 82.92% | 66.08% | 81.13% [0, 1, 2] | 78.03% [3, 4, 5] | [7, 11, 19, 16, 6, 13] |
| 99 | 86.50% | 66.08% | 77.38% [4, 2, 0] | 81.36% [1, 3, 5] | [11, 14, 13, 20, 11, 3] |
| 42 | 88.50% | 71.60% | 83.26% [0, 4, 1] | 75.53% [2, 3, 5] | [15, 6, 12, 12, 21, 6] |
| 17 | 85.67% | 80.07% | 76.37% [4, 2, 3] | 78.89% [0, 1, 5] | [5, 24, 14, 7, 13, 9] |
<p align="center">
  Table I. Accuracy Results from Varying Split (goal state of 24 samples & 6 bins)
</p>

Based on the result from Table I, the Markov Decision Process method performance heavily relies on how the data is split. The random states of 10, 50 and 42 result in a higher top 3 bins accuracy than bottom 3 bins, the top 3 bins accuracy is relatively close to the full training accuracy except for random state 42. Perform a similar test as above, but instead change the goal state to 36, the number of iteration to 20,000,000 and the number of bins to 4. The results from changiner the goal state and number of bins is shown in Table II.

| Split Random State | Full Training Accuracy | Naive Occupancy Accuracy | Top 2 Bins Accuracy | Bottom 2 Bins Accuracy | Bin Distribution | 
| --- | --- | --- | --- | --- | --- |
| 10 | 87.71% | 59.96% | 88.41% [0, 2] | 77.67% [1, 3] | [29, 16, 15 ,12] |
| 50 | 82.92% | 66.08% | 81.96% [1, 0] | 77.82% [2, 3] | [18, 16, 19 ,19] |
| 99 | 86.50% | 66.08% | 81.96% [1, 0] | 77.82% [2, 3] | [15, 13, 19 ,25] |

<p align="center">
  Table I. Accuracy Results from Varying Split (goal state of 36 samples & 4 bins)
</p>


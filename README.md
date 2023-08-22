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

Then the solution bins distribution is used to create a pool of data to train the occupancy model. For each solution bin the number of samples determines the number of sample randomly drawn from the corresponding total bin. The solution occupancy model accuracy therefore will vary depending on the sample drawn from each of the total bins. This method of applying the solution bins distribution take into account the variation in the model accuracy contributed by a sample in each bin. Apply the distribution to the training pool based on the solution bins for 1000 iteration and compute the mean, max and min of the resulting occupancy model. The resulting mean accuracy respresent the the likely accuracy of the occupancy model when the training data follow similar distribution to the solution bins.

## Result
The Q-learning algorithm is ran for 10,000,000 iterations with a goal state of 24 samples which take about 3 minutes and 20 seconds to run. However, the amount of avaliable data is limited, therefore testing different split of the data is necessary to see if this impact the solution. The results from varying the random_state of the train_test_split function from sklearn, while keeping K-mean random state the same is shown in Table I.

| Split Random State | Full Training Accuracy | Naive Occupancy Accuracy | Solution Average Accuracy | Solution Max Accuracy | Solution Min Accuracy | Total Bins Distribution | Solution Bins Distribution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 87.71% | 59.96% | 85.05% | 90.13% | 70.53% | [20, 14, 15, 2, 10, 11] | [5, 7, 5, 2, 0, 5] |
| 50 | 82.92% | 66.08% | 81.98% | 84.43% | 72.13% | [7, 11, 19, 16, 6, 13] | [7, 5, 2, 6, 4, 0] |
| 99 | 86.50% | 66.08% | 84.27% | 87.07% | 78.39% | [11, 14, 13, 20, 11, 3] | [5, 2, 6, 3, 6, 2] |
| 42 | 88.50% | 71.60% | 87.17% | 89.59% | 81.23% | [15, 6, 12, 12, 21, 6] | [10, 5, 4, 1, 4, 0] |
| 17 | 85.67% | 80.07% | 83.44% | 89.73% | 70.02% | [5, 24, 14, 7, 13, 9] | [5, 1, 6, 4, 6, 1] |
<p align="center">
  Table I. Accuracy Results from Varying Split (goal state of 24 samples & 6 bins)
</p>

Based on the result from Table I, the way the data is split does have a noticeable effect on the occupancy model accuracy and the accuracy of the solution. However, the Markov Decision Process method average solution accuracy only have a 1-3% difference from the full training accuracy across different data split. In the case for random state 10, 50 and 42, the solution bins distribution eliminate bin 5, 6 and 6 respectively with only a 1-2% drop in accuracy. Preform a similar test as above, but instead change the goal state to 36 and the number of bins to 4 to see the effect of reducing the amount of bins and increasing goal state sample. The new training parameter result in similar training time of around 3 minutes and 20 seconds. The results from changing the goal state and number of bins is shown in Table II.

| Split Random State | Full Training Accuracy | Naive Occupancy Accuracy | Solution Average Accuracy | Solution Max Accuracy | Solution Min Accuracy | Total Bins Distribution | Solution Bins Distribution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 87.71% | 59.96% | 83.63% | 90.05% | 71.68% | [29, 16, 15 ,12] | [18, 5, 11, 2] |
| 50 | 82.92% | 66.08% | 82.50% | 84.80% | 78.78% | [18, 16, 19 ,19] | [14, 15, 7, 0] |
| 99 | 86.50% | 66.08% | 84.72% | 87.14% | 79.23% | [15, 13, 19 ,25] | [10, 13, 10, 3] |
| 42 | 88.50% | 71.60% | 87.66% | 90.12% | 80.90% | [16, 24, 13, 19] | [11, 6, 13, 6] |
| 17 | 85.67% | 80.07% | 81.01% | 88.64% | 71.02% | [5, 29, 24, 14] | [1, 26, 2, 7] |

<p align="center">
  Table II. Accuracy Results from Varying Split (goal state of 36 samples & 4 bins)
</p>

Similar to the result from Table I, how the data is split influences the accuracy even more so in this case where the difference between the full training and average soluion accuracy is up to 4%. It is expected to have varying results due to the fact that the dataset used is very small and certain partition can have great effect. However, there seem to be no significant difference between using 4 bins and 6 bins. Next increase the number of bins to 8, change the goal state to 16 and change the number of iterations to 15,000,000 (the increase is required to get a consistence non zero solution). Then preform a similar test as above in order to see the effect of increasing the number of bins and decreasing goal state sample. The new training parameter result in a training time of around 4 minutes and 40 seconds. The result from changing the goal state and number of bins is shown in Table III.

| Split Random State | Full Training Accuracy | Naive Occupancy Accuracy | Solution Average Accuracy | Solution Max Accuracy | Solution Min Accuracy | Total Bins Distribution | Solution Bins Distribution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 87.71% | 59.96% | 82.26% | 90.08% | 64.61% | [20, 14, 2 ,4, 9, 11, 11, 1] | [2, 5, 0, 3, 1, 1, 3, 1] |
| 50 | 82.92% | 66.08% | 80.41% | 83.82% | 73.11% | [2, 10, 14 ,14, 9, 6, 7, 10] | [2, 4, 0, 0, 3, 2, 3, 2] |
| 99 | 86.50% | 66.08% | 83.43% | 86.47% | 79.99% | [9, 11, 6, 12, 8, 10, 13, 3] | [2, 3, 2, 3, 1, 2, 1, 2] |
| 42 | 88.50% | 71.60% | 85.67% | 88.84% | 78.43% | [7, 7, 13, 11, 6, 8, 12, 8] | [3, 0, 2, 2, 2, 2, 3, 2] |
| 17 | 85.67% | 80.07% | 83.76% | 90.44% | 67.71% | [5, 10, 10, 6, 8, 13, 13, 7] | [2, 2, 1, 3, 2, 2, 1, 3] |

<p align="center">
  Table III. Accuracy Results from Varying Split (goal state of 16 samples & 8 bins)
</p>

Like the previous test, the variation in how the data is split influences the difference between full training and average soulution accuracy which is between 2%-5%. From the limited tests from Table I-III, having an a "good" number of bins and goal state samples can slightly improve the solution accuracy compared to the full training accuracy for different splits of the data. However, what is consider as good parameters cannot be quantitatively determine. Another important part of this method is the K-mean clustering which could result in different grouping of data samples. Preform a similar test with a goal state samples of 24 and 6 number of bins, but fix the data split random state and instead vary the K-mean random initial state. The results from varying the random initial state of the K-mean function from sklearn is shown in Table IV.

| K-means Random State | Full Training Accuracy | Naive Occupancy Accuracy | Solution Average Accuracy | Solution Max Accuracy | Solution Min Accuracy | Total Bins Distribution | Solution Bins Distribution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 87.71% | 59.96% | 85.50% | 90.06% | 66.80% | [11, 12, 23, 10, 12, 4] | [4, 7, 4, 0, 5, 4] |
| 34 | 87.71% | 59.96% | 84.43% | 89.58% | 69.71% | [13, 12, 12, 21, 10, 4] | [2, 5, 7, 7, 0, 3] |
| 55 | 87.71% | 59.96% | 85.30% | 90.21% | 68.49% | [12, 12, 11, 10, 23, 4] | [4, 7, 2, 3, 4, 4] |
| 91 | 87.71% | 59.96% | 83.31% | 90.01% | 67.92% | [14, 8, 4, 7, 13, 26] | [5, 8, 4, 1, 3, 3] |
| 73 | 87.71% | 59.96% | 83.89% | 89.68% | 70.30% | [10, 24, 4, 12, 11, 11] | [2, 3, 4, 5, 6, 4] |
<p align="center">
  Table IV. Accuracy Results from Varying Split (goal state of 24 samples & 6 bins)
</p>

Based on Table IV. results, the K-means clustering on the data changes the difference between full training and average solution accuracy is up to 5%. The two important factors in the solution accuracy are bins grouping and how the data is split, although the latter may influences the clustering of the bins. The variation in splitting the data can just be due to the limited amout of data and having more data samples from different blocks can reduce this problem. However, the grouping of the bins still remain a major factor that determine the performance of this method. The K-means algorithm does not guaranteed the most optimal solution and the characteristics used to group the bins can not be determined. Another approach to group the bins can be looking at the correlation between blocks occupancy.

## Conclusion
In conclusion, this is no way a comprehesive testing of the method demonstrated above. The performance of this method is only tested using a ridge regression with a Gaussian kernel model, and being able to achieve similar results using other predicitive model is unknown. However, based on the limited tests this solution method for solving the parking occupancy multi-arms bandit problem shows that applying the solution bins distribution result in only a 1-3% difference between the full training and solution accuracy under the "best" conditions. The worst performance of this method based on the limited testing is up to a 5% difference between full training and solution accuracy. The solution distribution determines which bins and its corresponding data samples are most important in improving the model accuracy. In some cases entire bin can be omitted without significantly decreasing the model accuracy. Based on the solution distribution data collector can plan where to spent most of their resources to collect occupancy data and effeciently train their model.





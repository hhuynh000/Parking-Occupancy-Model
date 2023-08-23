# Parking-Occupancy-Model
## Introduction
Curbside parking occupancy is an important aspect of parking management for cities. However, training models to predict every blocks in a city would require collecting a lots of ground truth data. The collection of such data would cost time and money. This page explore the idea of finding blocks that are most meaningful for the model to learn from, therefore can potentially reduce the amount data require to be collected. 

## Background
The problem of finding the most meaningful blocks to train a model can treated as a type of multi-arms bandit problem. A collection of blocks can be clustered into groups with similar occupancy characteristics and each group can be considered as a bin. Each bin would give the model one data sample from the corresponding group of blocks to train in a predetermined order. The goal of the model agent is to maximize the prediction accuracy on a separate testing data. A solution is to represent the entire bins of data sample system as a Markov Decision Process, where each state is a combination of block samples from all the different bins. The action is therefore adding and removing block sample from one of the bin. The optimal solution can be found using Q-learning algorithm. After training, the resulting solution bins distribution tells us which bin of block samples is more impactful in improving the model accuracy. A sample Markov Decision Process state space for 2 bins with 2 data samples each is shown in Figure 1 below.

<p align="center">
  <img src="https://github.com/hhuynh000/Parking-Occupancy-Model/blob/main/figures/state_space.png" width="400"/>
</p>
<p align="center">
  Figure 2. K-mean Clustering on Naive Parking Occupancy
</p>

## Implementation
The parking occupancy data used is gathered and processed from Seattle Department of Transportation (SDOT). The naive and ground truth data consist of 12 different blocks around Seattle, where each block has data collected for 7 day (03/21/22 - 03/26/22 & 03/28/22) from 8am to 8pm with a time interval of 1 minute. In total there are 84 whole day of parking occupancy data from 8am to 8pm across the 12 blocks and 7 days for each block. The data is split into a training and testing set where there are 72 whole day of training and 12 whole day of testing data. The occupancy data is normalized as an occupancy percentage based on the max parking capacity of each individual blocks.

The base predictive model used is a ridge regression with a radial basis function kernel. This particular model is chosen because it is simple to train with little data samples, while still having decent accuracy. The sklearn kernel_ridge package is used to implement the model with lambda value of 0.01 and gamma value of 0.001.  The accuracy is computed using the mean absolute error:

$$ Accuracy = 1 - \frac{ \sum\limits_{i=0}^n \vert y_{i} - x_{i} \vert }{n} $$


K-mean clustering is used to group the blocks into 6 individual bins. The number of bin is chosen arbitary, the number 6 is used because it is the number of different sub-area the 12 blocks reside in: University District, Belltown, Uptown Triangle, Denny Triangle, Pike-Pine and South Lake Union. There are no definitive correlation between the parking occupancy and sub-area of the blocks, however this initial assumption is made in order to derive the number of bin to use. The training data are clustered using K-mean into 6 different bins. One of the resulting clusters when mapped on to the first 2 principal components is shown in Figure 2. Then every training data sample get assigned to there corresponding bin based on the result from K-mean clustering.

<p align="center">
  <img src="https://github.com/hhuynh000/Parking-Occupancy-Model/blob/main/figures/K-mean.png" width="600"/>
</p>
<p align="center">
  Figure 2. K-mean Clustering on Naive Parking Occupancy
</p>

After grouping the data samples into there corresponding bin, the bins are setup as state in the Markov Decision Process using the class BanditMDP. The initial state of the prediction model in the Markov Decision Process is having zero data sample ([0,0,0,0,0,0]). Also, The model requires a goal state to be set, which indicate the final total number of data samples from all bins used to train the model. A smaller goal state will take in account less of the overall data but the solution is faster to find, whereas a larger goal state will take into account more of the data but the solution will be slower to find. When the model reaches a goal state, the model can "exit" and claim a reward which is equal to the model accuracy when tested on the testing data. After the model reaches the goal state the model will start over with zero block. At every step the model can add or remove a data point from the data pool being used to train the model. The reward given at any non "exit" action is -0.01 in order to discourage the model from repeating state it already been to. 

The Q-learning algorithm is used train the model into finding the optimal combination of data sample from each bin which will produce the highest accuracy. The algorithm is an iterative process to find Q-value as the model explore the state space. The new sample estimate of Q-value given a sample of the current state s, the action a, the next state s' and the reward r (s,a,s',r) is described in Equation 1. The new updated Q-value incorporate the new estimate into a running average describes by Equation 2, where alpha is the percentage of the new estimate to update the Q-value. The default $\alpha$ value used is 0.9.

$$ sample = R(s,a,s') + \gamma \max\limits_{a'} Q(s',a') \quad \quad [1] $$

$$ Q(s,a) = (1-\alpha)Q(s,a) + \alpha (sample) \quad \quad [2] $$

In order to encourage exploration, an epsilon greedy function is used to incorporate a possibility of a random action taken by the model. There is a probablity of $\epsilon$ to perform a random action based on the current state and there is a probablity of $1-\epsilon$ of acting based on the current policy. The default $\epsilon$ value used is 0.5 to encourage exploration because the only big reward is at the goal states. The implementation of the Markov Decision Process and Q-Learning Algorithm can be found in the "MDP.py" file.

Then the solution bins distribution is used to create a pool of data to train the occupancy model. For each solution bin the number of samples determines the number of samples randomly drawn from the corresponding total bin. The solution occupancy model accuracy therefore will vary depending on the sample drawn from each of the total bins. This method of applying the solution bins distribution takes into account the variation in data sample from a bin contribution in improving the accuracy of the model. Apply the distribution to the solution training pool based on the solution bins for 1000 iterations and compute the mean, max and min of the resulting occupancy model. The resulting mean accuracy respresent the likely accuracy of the occupancy model when the training data follow similar distribution to the solution bins.

## Result
All the tests and results can be seen in the Jupyter notebook file called "MDP_Test.ipynb".

The avaliable data set is limited in its size, therefore testing different split of the data is necessary to see if this impact the solution. The Q-learning algorithm is ran for 10,000,000 iterations with a goal state of 24 samples and 6 bins which take about 3 minutes and 40 seconds to run. The results from varying the random_state of the train_test_split function from sklearn package, while keeping K-mean random state fixed is shown in Table I.

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

Based on the result from Table I, the way the data is split does have a noticeable effect on the occupancy model accuracy and the accuracy of the solution. However, the average solution accuracy only have a 1-3% difference from the full training accuracy across different data split. In the case for random state 10, 50 and 42, the solution bins distribution eliminate bin 5, 6 and 6 respectively with only a 1-2% drop in accuracy. Perform a similar test as above, but instead change the goal state to 36 and the number of bins to 4 to see the effect of reducing the amount of bins and increasing goal state sample. The new training parameters result in similar training time of around 3 minutes and 40 seconds. The results from changing the goal state and number of bins is shown in Table II.

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

Similar to the result from Table I, how the data is split influences the accuracy even more so in this case where the difference between the full training and average soluion accuracy is up to 5%. It is expected to have varying results due to the fact that the dataset used is very small and certain partition can have great effect. Next increase the number of bins to 8, change the goal state to 16 and change the number of iterations to 15,000,000 (the increase is required to get a consistence non zero solution). Then preform a similar test as above in order to see the effect of increasing the number of bins and decreasing goal state sample. The new training parameters result in a training time of around 4 minutes and 40 seconds. The result from changing the goal state and number of bins is shown in Table III.

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

Like the previous test, the variation in how the data is split influences the difference between full training and average soulution accuracy which is between 2%-6%. From the limited tests from Table I-III, having an a "good" number of bins and goal state samples can slightly improve the solution accuracy compared to the full training accuracy for different splits of the data. However, what is consider as good parameters cannot be quantitatively determine. Another important part of this method is the K-mean clustering which could result in different grouping of data samples. Perform a similar test with a goal state samples of 24 and 6 number of bins, but fix the data split random state and instead vary the K-mean random initial state. The results from varying the random initial state of the K-mean function from sklearn package is shown in Table IV.

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

Based on Table IV. results, the K-means clustering on the data changes the difference between full training and average solution accuracy is up to 5%. The two important factors in the solution accuracy are bins grouping and how the data is split, although the latter may influences the clustering of the bins. The variation in splitting the data can just be due to the limited amout of data and having more data samples from different blocks can reduce this problem. However, the grouping of the bins still remain a major factor that determine the performance of this method. The K-means algorithm does not guaranteed the most optimal solution and the characteristics used to group the bins cannot be determined. Another approach to group the bins can be looking at the correlation between blocks occupancy.

## Conclusion
In conclusion, this is no way a comprehesive testing of the method demonstrated above. The performance of this method is only tested using a ridge regression with a radial basis function kernel model, and being able to achieve similar results using another predicitive model is unknown. However, based on the limited tests this solution method for solving the parking occupancy multi-arms bandit problem shows that applying the solution bins distribution result in only a 1-3% difference between the full training and solution accuracy under the "best" conditions. The worst performance of this method based on the limited testing is up to a 5% difference between full training and solution accuracy. The solution distribution determines which bins and its corresponding data samples are most important in improving the model accuracy. In some cases entire bin can be omitted without significantly decreasing the model accuracy. Based on the solution distribution data collector can plan where to spent most of their resources to collect occupancy data and effeciently train their model.

The issues of this method are that the grouping of bins using K-means does not guranteed an optimal solution and the training time is relatively long, around 3 minutes and 40 seconds just for 72 data samples with 6 bins. Another approach to group the bins could be using correlation between blocks occupancy, which might be better at filtering blocks occupancy characteristics instead of K-means clustering. The training time when scaling up the data size can be addressed by partioning data to reasonable chunk size for training, then cross validate across the different chunk of data.





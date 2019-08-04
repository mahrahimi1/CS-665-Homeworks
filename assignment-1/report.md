#### Mushroom Dataset Summary:
- Number of original features: 22
- Number of *feature_value_pairs* (binary) features: 117
- Number of labels: 2
- Number of data points in train set: 4062
- Number of data points in test set: 2031

#### Tumor Dataset Summary:
- Number of original features: 17
- Number of *feature_value_pairs* (binary) features: 42
- Number of labels: 18 
- Number of data points in train set: 169
- Number of data points in test set: 85

&nbsp;
### Results:

We varied the complexity of our models by varying the depths of the trees and we obtained the training and test accuracy values as demonstrated below. There are two tables for the Mushroom dataset; one for the basic procedure (CIML) and one for the ID3 criterion. Similarly, there are two tables for the Tumor dataset. 

In the Mushroom dataset, as we increase the depth of the decision tree, the training and test accuracies become increasingly better and better until they reach 100 percent. In the ID3 decision tree, it even converges to 100 percent accuracy faster than the basic decision tree. However, we observe that over the Tumor dataset, the decision trees do not perform well compared to the Mushroom dataset. Initially, as we increase the depth of the trees over the Tumor dataset, the accuracies increse. This reaches its maximum approximately at depth=3. However, after that increasing the depth of the trees only decreses the accuracy. In general, the performance is lower than the performace over the Mushroom dataset. Apparently, the decision trees learn and capture the inherent structure of the Mushroom dataset better than the Tumor dataset. This could be because the Mushroom dataset has much more data points in its training set. As demonstrated at the top of this report, Mushroom training set has 4062 data points and Tumor training set has 169 data points. Furthermore, the Mushroom dataset not only has more examples, but also has less labels. It has only two labels and an abundance of training data points for each label. However, the Tumor dataset has 18 active labels and 169 data points in total. Therefore, the number of datapoints for each label is small. Moreover, the data points are not equally distributed between the 18 labels; there are few datapoints for some of those labels. As a result, the Tomor dataset may not be providing sufficient number of datapoints for the decision trees to be able to generalize well.


&nbsp;
#### Mushroom Dataset (Basic):

| Depth | Train Accuracy | Test Accuracy |
|:-----:|:--------------:|:-------------:|
|   1   |     88.68%     |     88.04%    |
|   3   |     98.50%     |     98.52%    |
|   5   |     99.90%     |     99.80%    |
|   10  |     99.98%     |     99.90%    |
|   20  |     100.00%    |    100.00%    |
|   50  |     100.00%    |    100.00%    |

#### Mushroom Dataset (ID3):

| Depth | Train Accuracy | Test Accuracy |
|:-----:|:--------------:|:-------------:|
|   1   |     88.68%     |     88.04%    |
|   3   |     96.33%     |     96.41%    |
|   5   |     100.00%    |    100.00%    |
|   10  |     100.00%    |    100.00%    |
|   20  |     100.00%    |    100.00%    |
|   50  |     100.00%    |    100.00%    |


#### Tumor Dataset (Basic):

| Depth | Train Accuracy | Test Accuracy |
|:-----:|:--------------:|:-------------:|
|   1   |     29.59%     |     31.76%    |
|   3   |     43.20%     |     41.18%    |
|   5   |     51.48%     |     36.47%    |
|   10  |     47.34%     |     38.82%    |
|   20  |     41.42%     |     30.59%    |
|   50  |     37.87%     |     32.94%    |

#### Tumor Dataset (ID3):

| Depth | Train Accuracy | Test Accuracy |
|:-----:|:--------------:|:-------------:|
|   1   |     29.59%     |     31.76%    |
|   3   |     42.60%     |     36.47%    |
|   5   |     53.85%     |     31.76%    |
|   10  |     48.52%     |     29.41%    |
|   20  |     41.42%     |     27.06%    |
|   50  |     36.69%     |     32.94%    |


<br><br>
The implementation was done with Python 3.6.4.

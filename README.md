Enron Email project by Amishi


### Overview

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives.

In this project I will build a person of interest(POI) identifier based on financial and email data made public as a result of the Enron scandal. I used email and financial data for 146 executives at Enron to identify persons of interest in the fraud case. A person of interest (POI) is someone who was indicted for fraud, settled with the government, or testified in exchange for immunity. This report documents the machine learning techniques used in building a POI identifier.


### Data Exploration

Enron Submission Free-Response Questions :
1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?

## Number of data points

146

## Features

The dataset includes two categories of feature, namely financial features and email features.
Total number of persons in the data set: 146
Total number of persons of interest (poi) in the data set : 18
Total number of non persons of interest (non poi) in the data set : 128
Each person has 21 features

Financial features:  'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'.

Email features: 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'fraction_to_poi', 'fraction_from_poi'.

## POI

POI is the lable in the dataset represents people who has committed fraud. There are 18 data points with POI=1.

## Outliers

As I checked the dataset, I removed two outliers.

One is 'TOTAL'. Because this data point is the total number of all the other data points. Also while scanning through the dataset I found a datapoint named "THE TRAVEL AGENCY IN THE PARK" . As it was no way related to the employees I removed .

## Missing values

With removing outliers and add features, I count the missing values for each feature. The result is as the follows.

Total number of missing values in features: 1358

('salary', 51)
('to_messages', 60)
('deferral_payments', 107)
('total_payments', 21)
('long_term_incentive', 80)
('loan_advances', 142)
('bonus', 64)
('restricted_stock', 36)
('restricted_stock_deferred', 128)
('total_stock_value', 20)
('shared_receipt_with_poi', 60)
('from_poi_to_this_person', 60)
('exercised_stock_options', 44)
('from_messages', 60)
('other', 53)
('from_this_person_to_poi', 60)
('deferred_income', 97)
('expenses', 51)
('email_address', 35)
('director_fees', 129)

2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that doesn’t come ready-­made in the dataset-­-­explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) If you used an algorithm like a decision tree, please also give the feature importances of the features that you use.¶

In order to select features, I used Decision Tree as a sample model for multiple iterations. 

DecisionTreeClassifier(compute_importances=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, min_density=None, min_samples_leaf=1, min_samples_split=2, random_state=None, splitter='best') Accuracy: 0.81827 Precision: 0.31346 Recall: 0.30500 F1: 0.30917 F2: 0.30666 Total predictions: 15000 True positives: 610 False positives: 1336 False negatives: 1390 True negatives: 11664

salary : 0.0 deferral_payments : 0.0 total_payments : 0.0833333333333 loan_advances : 0.0 bonus : 0.0 restricted_stock_deferred : 0.0 deferred_income : 0.0 total_stock_value : 0.0907738095238 expenses : 0.187971552257 exercised_stock_options : 0.296613945578 other : 0.0 long_term_incentive : 0.0 restricted_stock : 0.0331909937888 director_fees : 0.0 to_messages : 0.0 from_poi_to_this_person : 0.047619047619 from_messages : 0.0860119047619 from_this_person_to_poi : 0.0 shared_receipt_with_poi : 0.047619047619 fraction_to_poi : 0.126866365519 fraction_from_poi : 0.0

##For first iteration above, we input all features into the model. 9 features in the model are important (importance > 0). They are: 'total_payments', 'total_stock_value', 'expenses', 'exercised_stock_options', 'restricted_stock', 'from_poi_to_this_person', 'from_messages','shared_receipt_with_poi','fraction_to_poi'.

DecisionTreeClassifier(compute_importances=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, min_density=None, min_samples_leaf=1, min_samples_split=2, random_state=None, splitter='best') Accuracy: 0.83173 Precision: 0.35131 Recall: 0.30950 F1: 0.32908 F2: 0.31705 Total predictions: 15000 True positives: 619 False positives: 1143 False negatives: 1381 True negatives: 11857

total_payments : 0.0833333333333 total_stock_value : 0.0302197802198 expenses : 0.123685837972 exercised_stock_options : 0.357167974882 restricted_stock : 0.0331909937888 from_poi_to_this_person : 0.0 from_messages : 0.0860119047619 shared_receipt_with_poi : 0.111904761905 fraction_to_poi : 0.174485413138

##Using 9 important features into the Decision Tree model, second iteration is shown above. Except the feature 'from_poi_to_this_person' of which importance equals to 0, other features are important.

DecisionTreeClassifier(compute_importances=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, min_density=None, min_samples_leaf=1, min_samples_split=2, random_state=None, splitter='best') Accuracy: 0.82607 Precision: 0.31886 Recall: 0.26800 F1: 0.29123 F2: 0.27683 Total predictions: 15000 True positives: 536 False positives: 1145 False negatives: 1464 True negatives: 11855

total_payments : 0.047619047619 total_stock_value : 0.0302197802198 expenses : 0.159400123686 exercised_stock_options : 0.357167974882 restricted_stock : 0.0808100414079 from_messages : 0.0860119047619 shared_receipt_with_poi : 0.111904761905 fraction_to_poi : 0.126866365519

##With 8 important features from the second iteration, the third iteration above run the decision tree model again, proving all the eight features selected are important.

##As a result, I selected 8 features to build my machine learning models. They are: 'total_payments', 'total_stock_value', 'expenses', 'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi', 'from_messages','fraction_to_poi'.
 
##Among these features, 'fraction_to_poi' and 'fraction_from_poi' are generated from existed email related variables. 'fraction_to_poi' equals 'from_this_person_to_poi' divided by 'from_messages'. 'fraction_from_poi' equals 'from_poi_to_this_person' divided by 'to_messages'. These added feature could interpret the relationship between persons and poi better than existed variables. Otherwise, people with small total number of mails will not show up through our model. So I added these two new features and expected them to represent email features.

##I used Decision tree to select the important features. .When there is any algorithm that computes distance or assumes normality, scale your features. Standardscaler for scaling purpose which were then feed to the KNN classifier.Scaling is critical, while performing Principal Component Analysis(PCA). PCA tries to get the features with maximum variance and the variance is high for high magnitude features. This skews the PCA towards high magnitude features.Tree based models are not distance based models and can handle varying ranges of features. Hence, Scaling is not required while modelling Decision trees.

3. What algorithm did you end up using? What other one(s) did you try?

### Decision Tree

DecisionTreeClassifier(compute_importances=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, min_density=None, min_samples_leaf=1, min_samples_split=2, random_state=None, splitter='best') Accuracy: 0.82607 Precision: 0.31886 Recall: 0.26800 F1: 0.29123 F2: 0.27683 Total predictions: 15000 True positives: 536 False positives: 1145 False negatives: 1464 True negatives: 11855

total_payments : 0.047619047619 total_stock_value : 0.0302197802198 expenses : 0.159400123686 exercised_stock_options : 0.357167974882 restricted_stock : 0.0808100414079 from_messages : 0.0860119047619 shared_receipt_with_poi : 0.111904761905 fraction_to_poi : 0.126866365519

### Gaussian Naive Bayes

GaussianNB() Accuracy: 0.84807 Precision: 0.38867 Recall: 0.24350 F1: 0.29942 F2: 0.26316 Total predictions: 15000 True positives: 487 False positives: 766 False negatives: 1513 True negatives: 12234

### K-Nearest Neighbors

Pipeline(steps=[('scale', StandardScaler(copy=True, with_mean=True, with_std=True)), ('knn', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_neighbors=5, p=2, weights='uniform'))]) Accuracy: 0.86280 Precision: 0.43318 Recall: 0.09400 F1: 0.15448 F2: 0.11145 Total predictions: 15000 True positives: 188 False positives: 246 False negatives: 1812 True negatives: 12754

### Random Forest

RandomForestClassifier(bootstrap=True, compute_importances=None, criterion='gini', max_depth=None, max_features='auto', max_leaf_nodes=None, min_density=None, min_samples_leaf=1, min_samples_split=2, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0) Accuracy: 0.85427 Precision: 0.35514 Recall: 0.11400 F1: 0.17260 F2: 0.13191 Total predictions: 15000 True positives: 228 False positives: 414 False negatives: 1772 True negatives: 12586

total_payments : 0.132165501822 total_stock_value : 0.0842622479555 expenses : 0.12935687862 exercised_stock_options : 0.191842369768 restricted_stock : 0.11368767859 from_messages : 0.0333290762057 shared_receipt_with_poi : 0.162363356029 fraction_to_poi : 0.15299289101


4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well? How did you tune the parameters of your particular algorithm? (Some algorithms don’t have parameters that you need to tune-­-­if this is the case for the one you picked, identify and briefly explain how you would have done it if you used, say, a decision tree classifier).

As we saw before although the accuracy was high but the precision and recall score was zero in most cases. This means the accuracy, precision, recall or other performance measure is not as good as thay could be if the model is not customized to the particular dataset’s features . Thus to improve the precision and recall score for all the classifiers we need to tune the parameters to optimize the classification models to give us a best possible result.

In my case I first mannually tried different algorithms with different combinations of parameters .But then I found it really useful using Grid search.


Manual tuning included determining which parameters to add to each algorithm and adding/removing features. GridSearchCV provided a convenient way to perform linear combinations for all of the different parameters and report the best result.

In a nutshell, parameter tuning is important because it optimizes our algorithm's performance on our given data set. To measure our algorithm's performance, we need to validate and evaluate our data for each different combination of our selected parameters. Algorithms tend to be very general in nature and not specifically tuned to any data set. Therefore, we must iteratively tune our algorithm until we obtain an evaluation we are satisified with.

### Decision Tree Tuning
GridSearchCV(cv=None, estimator=DecisionTreeClassifier(compute_importances=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, min_density=None, min_samples_leaf=1, min_samples_split=2, random_state=None, splitter='best'), fit_params={}, iid=True, loss_func=None, n_jobs=1, param_grid={'splitter': ('best', 'random'), 'criterion': ('gini', 'entropy')}, pre_dispatch='2*n_jobs', refit=True, score_func=None, scoring='recall', verbose=0) Accuracy: 0.82360 Precision: 0.32709 Recall: 0.30550 F1: 0.31593 F2: 0.30959 Total predictions: 15000 True positives: 611 False positives: 1257 False negatives: 1389 True negatives: 11743

best_params

{'splitter': 'random', 'criterion': 'entropy'} DecisionTreeClassifier(compute_importances=None, criterion='entropy', max_depth=None, max_features=None, max_leaf_nodes=None, min_density=None, min_samples_leaf=1, min_samples_split=2, random_state=None, splitter='random') Accuracy: 0.82680 Precision: 0.34667 Recall: 0.33800 F1: 0.34228 F2: 0.33970 Total predictions: 15000 True positives: 676 False positives: 1274 False negatives: 1324 True negatives: 11726

total_payments : 0.0973186272037 total_stock_value : 0.0625 expenses : 0.111317475172 exercised_stock_options : 0.18530624806 restricted_stock : 0.0808444264639 from_messages : 0.0363532823221 shared_receipt_with_poi : 0.13678033831 fraction_to_poi : 0.289579602468

This part is to tune original Decision Tree model on the parameters of criterion and splitter. From this tuning process, the model is optimized with splitter as 'random' and criterion as 'entropy'. The Recall of this model is improved to 0.338 and Precision to 0.347.


### KNN Tuning
GridSearchCV(cv=None, estimator=Pipeline(steps=[('scale', StandardScaler(copy=True, with_mean=True, with_std=True)), ('knn', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_neighbors=5, p=2, weights='uniform'))]), fit_params={}, iid=True, loss_func=None, n_jobs=1, param_grid={'knn__algorithm': ('ball_tree', 'kd_tree', 'brute', 'auto'), 'knn__n_neighbors': [1, 8]}, pre_dispatch='2*n_jobs', refit=True, score_func=None, scoring='recall', verbose=0) Accuracy: 0.85347 Precision: 0.43210 Recall: 0.31500 F1: 0.36437 F2: 0.33305 Total predictions: 15000 True positives: 630 False positives: 828 False negatives: 1370 True negatives: 12172

best_params
{'knn__algorithm': 'ball_tree', 'knn__n_neighbors': 1} Pipeline(steps=[('scale', StandardScaler(copy=True, with_mean=True, with_std=True)), ('knn', KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, metric='minkowski', metric_params=None, n_neighbors=1, p=2, weights='uniform'))]) Accuracy: 0.85347 Precision: 0.43210 Recall: 0.31500 F1: 0.36437 F2: 0.33305 Total predictions: 15000 True positives: 630 False positives: 828 False negatives: 1370 True negatives: 12172

This part is to tune original KNN model on the parameters of knn_algorithm and knn_n_neighbors. From this tuning process, the model is optimized with knn_algorithm as 'ball_tree' and knn_n_neighbors as 1. The Recall of this model is improved to 0.315 and Precision to 0.432


## Result

Decision Tree (Tuned) | 0.347 | 0.338
KNN (Tuned) | 0.432 | 0.315
Gaussian Naive Bayes | 0.389 | 0.244
Random Forest | 0.355 | 0.114

From this result, I select Decision Tree as the final algorithm considering performance both on precision and recall.

The specific algorithm shows below.

DecisionTreeClassifier(compute_importances=None, criterion='entropy', max_depth=None, max_features=None, max_leaf_nodes=None, min_density=None, min_samples_leaf=1, min_samples_split=2, random_state=None, splitter='random') Accuracy: 0.82680 Precision: 0.34667 Recall: 0.33800 F1: 0.34228 F2: 0.33970 Total predictions: 15000 True positives: 676 False positives: 1274 False negatives: 1324 True negatives: 11726

total_payments : 0.0973186272037 total_stock_value : 0.0625 expenses : 0.111317475172 exercised_stock_options : 0.18530624806 restricted_stock : 0.0808444264639 from_messages : 0.0363532823221 shared_receipt_with_poi : 0.13678033831 fraction_to_poi : 0.289579602468


5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?

For the chosen algorithm, we need to validate it to see how well the algorithm generalizes beyond the training dataset. A classic mistake we might make is to use same dataset for training and testing.When evaluating different parameters for estimators, there was still a risk of overfitting on the test set because the parameters can be tweaked until the estimator performs optimally. This way, knowledge about the test set can “leak” into the model and evaluation metrics no longer report on generalization performance. To solve this problem, yet another part of the dataset can be held out as validation set. 

The whole dataset we have includes only 146 data points, which is very small. So I chose stratified shuffle split cross validation to validate the selected algorithm.

6. Give at least 2 evaluation metrics, and your average performance for each of them. Explain an interpretation of your metrics that says something human-­ understandable about your algorithm’s performance.

### Precision

0.338
precision = number of true positive / (number of true positive + number of false positive) i.e the percentage of POI identified correctly out of all identified POI .In my case it is 0.5 which means 50% my model will predict a correct POI.

Precision indicated that 34% of people who are predicted as poi are truly people of interests.

### Recall

0.347
recall = number of true positive / (number of true positive + number of false negative) i.e the probability that the model flags a POI when it is actually a POI .In my case it is 0.4 which means 40% of time my model will predict a POI correctly if he is actually a POI .

Recall here indicated that among people of interests, 35% of them are correctly predicted via our final algorithm.





# sp-crush-enemies

# 1_data_and_model_exploration

In this part we have only used the train data set: 'Data/Regression_Supervised_Train.csv'

1. Explore data:
    - Checked the distribution of the target value:
        removed some outliers, log transformed it
    - Checked on NaN values:
        added educated guess, imputed, and removed columns with high percentage of NaN
    - Removed columns with binomial distribution and low variability
    - Create categories for `year` variable using k-means
    - Transform categorical variables to dummy columns
    - Normalization scaler for continuous variables.
    
2. Models exploration:
   Tested different models: decission tree, random forest and gradient boosting.
       
   - Varied depth of the trees to get the depth with best performance.
   - Tested different scaling on continuous variables to choose the one with best performance.
   - Explored the loss function to get better performance.

# 2_prediction_and_evaluation

1. Data processing for both, train and test set:
    - Train: 'Data/Regression_Supervised_Train.csv'
    - Test: 'Data/Regression_Supervised_Test_1.csv'

2. Apply selected model from previous model exploration:
    - Model: GradientBoostingRegressor(min_samples_split=100, max_depth=8, subsample=0.8, loss='huber')

3. Results:
    MAE: 197357.17947505237
    
    ![Results](/Data/results_test.png)

      

# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Tomás Baidal Soliveres

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?

When trying to submit our predictions, we have obtained some negative values. As Kaggle refuses the negative prediction values when submitting results, we have to replace those negative values with zeros.

We have submitted 3 different results of 3 different models:

- **FIRST MODEL:** *first raw submission* with the raw data provided. A total of 14 models were trained, using various techniques such as *StackerEnsembleModels* for different algorithms (e.g., *RandomForest, LightGBM, CatBoost*), *WeightedEnsembleModel*, and bagging. The process utilized multi-layer stack-ensembling (with 3 levels) and bagging (with 8 folds). **The best performing model, based on the validation score, was *WeightedEnsemble_L3* with a validation score of -53.091580**. In this case, we did not obtain negative values.

- **SECOND MODEL:** *new features* with an EDA and Feature Engineering applied. **The best-performing model is again the *WeightedEnsemble_L3*, with a validation score of approximately -35.71.** The fit time for this model was about 541.43 seconds, and it took about 14.84 seconds to make predictions on the validation set. Other trained models include *CatBoost_BAG_L2, WeightedEnsemble_L2, ExtraTreesMSE_BAG_L2, LightGBM_BAG_L2*, and others. These models differ in their prediction times, fit times, and validation scores. The raw data that was processed for model training consisted of eight categorical features and six integer (boolean) features. **The categorical features include 'weather', 'month', 'day', 'hour', 'dayofweek', etc. The integer features include 'holiday', 'workingday', 'year', 'sunlight_cat' and 'wind_cat'.*

- **THIRD MODEL:** *new features with hyperparameters*. **The model with the best performance was the *WeightedEnsemble_L2* model with a validation score of -35.86.** This model is an ensemble that uses a weighted combination of predictions from the individual models, which helps improve the overall performance. The validation score mentioned here is a negative number because AutoGluon always treats the problem as a maximization problem. Thus, for regression problems where you would generally minimize the error (like *MSE*), it takes the negative of the metric, hence the negative values. In the second run, models were trained at two stack levels, which is an example of multi-layer stack-ensembling. This strategy can lead to improved performance because predictions of base models are used as features for higher-level models. The fit times, predict times, and model paths have been provided for each model. For example, the WeightedEnsemble_L2 model, which is the best-performing model, took approximately 0.47 seconds to fit and 0.00097 seconds for prediction.

### What was the top ranked model that performed?

The top-ranked model according to the output summary is 'WeightedEnsemble_L2'.

The Weighted Ensemble method is a type of ensemble learning technique that works by combining the predictions from multiple different machine learning models. It assigns weights to each model's predictions based on their performance, so more accurate models have a larger influence on the final prediction.

This specific model 'WeightedEnsemble_L2' is a level-2 ensemble. This suggests that it's an ensemble of base models (level-1), probably including different types of models like LightGBM, XGBoost, CatBoost, RandomForest, and ExtraTrees.

The score of this model (score_val) is -35.860850, which is the highest (least negative) among all the models, implying that it had the best performance on the validation data. In this context, the score is probably a loss function where lower values are better, and negative values are possible depending on the specific loss function used.

The 'fit_time' for 'WeightedEnsemble_L2' is 177.479388, which means it took this amount of time to train the model, and 'pred_time_val' is 4.352890, which indicates the time it took to make predictions on the validation set.

'WeightedEnsemble_L2' is the second layer in a multi-layer stack ensembling setup, which means that its input features are the predictions of the models in the layer below (i.e., level-1 models). This type of approach can often lead to better performance as it can learn from the strengths and weaknesses of various base models.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?

The EDA we performed suggests that time-based features, temperature, 'atemp', humidity, and windspeed, as well as whether it's a working day, have a significant impact on the target variable, 'count'.

Based on these findings, new features were engineered. Here's what each one represents:

- **'year', 'month', 'day', 'hour':** These represent the year, month, day, and hour extracted from the datetime feature. It seems that these finer-grained time features might have a better correlation with the target variable than the full datetime.

- **'dayofweek':** This represents the day of the week, which can be useful because usage might vary between weekdays and weekends.

- **'hour_workingday_interaction':** This is an interaction term between the hour of the day and whether it's a working day. This could be useful because usage patterns might differ during working hours on working days compared to non-working hours or non-working days.

- **'sunlight_cat':** This categorizes the hour of the day into whether it's day or night, which could impact usage.

- **'temp_cat', 'atemp_cat':** These features categorize the temperature and 'feels-like' temperature into different categories, which could have a non-linear relationship with the usage.

- **'wind_cat':** This feature categorizes the wind speed, which could also have a non-linear relationship with usage.

- **'humid_category':** This feature categorizes the humidity, which could also have a non-linear relationship with usage.

- **The 'datetime', 'temp', 'atemp', 'humidity', 'windspeed', and 'season'** columns were dropped after these new features were created, as they had been effectively replaced by the new features.


### How much better did your model preform after adding additional features and why do you think that is?

The first model we built used raw features without much feature engineering. **The result was a Root Mean Squared Logarithmic Error (RMSLE) of 1.79080.**

After we added additional features and carried out feature engineering, **the RMSLE score improved dramatically to 0.49546. This is an approximate improvement of about 72.3% in terms of RMSLE.**

Here are a few reasons why the performance of the model could have improved significantly:

- **More Information:** By breaking down date-time into hour, day, month, and year, we provided the model with a more granular perspective on time. This allowed the model to identify any trends or patterns occurring at specific times of day, specific days of the week, or during specific months or years.

- **Interaction Features:** We introduced an interaction term between 'hour' and 'workingday'. This could have helped the model to understand complex relationships between these variables that could affect bike rental demand.

- **Categorical Features:** By categorizing variables such as temperature, humidity, windspeed, and apparent temperature (atemp), the model might have found it easier to delineate between different ranges of these variables, allowing it to recognize different patterns in the data that were not evident when these variables were treated as continuous.

- **Business Intuition:** We introduced a 'sunlight_cat' feature, which separates daytime from nighttime. This feature might capture the common-sense understanding that bike rental patterns would differ between day and night.

In conclusion,the new features that we created helped the model to better understand the nuances of the dataset, resulting in a more accurate model.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?

After tuning the hyperparameters, **our RMSLE score changed from 0.49546 to 0.52306.** This is actually an increase, meaning the performance of the model was slightly worse after hyperparameter optimization.

Typically, we expect hyperparameter tuning to improve model performance because it involves finding the optimal configuration for the model to learn from the data. However, it's not guaranteed to always improve results, and in some cases, it might even make them worse, as observed in this scenario.

Here are a few potential reasons why the model's performance may not have improved with hyperparameter tuning:

- **Overfitting to the validation set:** During hyperparameter tuning, it's possible that we end up overfitting to the validation set, especially if we are doing a lot of experiments and checking our validation score often. This could lead to a model that performs worse on the test set.

- **Randomness in the tuning process:** Depending on the method of hyperparameter tuning used, there could be an element of randomness in the results. For example, methods like random search or Bayesian optimization may not always find the absolute best parameters due to the stochastic nature of the process.

- **Noisy data or over-complex model:** If the data is noisy or the model is too complex, then tweaking hyperparameters may not lead to better performance on the test set. Sometimes, simpler models with fewer hyperparameters can perform better on such data.



### If you were given more time with this dataset, where do you think you would spend more time?

If given more time with this dataset, I would focus on several key areas to further enhance the performance and accuracy of the models:

- **Feature Engineering:** I would spend more time creating and experimenting with new features that could be relevant to the target variable. This could involve more complex transformations, interaction terms, or additional domain-specific insights.

- **Hyperparameter Tuning:** Although automated machine learning tools can be very efficient, there is often more performance that can be squeezed out through manual tuning of model hyperparameters. I would spend more time optimizing these parameters to get the best out of the selected models.

- **Ensemble Techniques:** Combining different models to create an ensemble often results in better performance than any individual model. I would experiment with various ensemble techniques to potentially improve our results.

- **Exploring More Models:** I would consider trying out different machine learning algorithms, including those not covered by the automated ML framework. This would provide a broader perspective and possibly better performance.

- **Data Cleaning and Preprocessing:** Revisiting the initial steps of the data processing pipeline to investigate if there are other data cleaning, transformation, or preprocessing steps that could impact the model performance.

- **Understanding the Errors:** I would spend more time understanding the specific instances where the model's predictions are incorrect. This could lead to insights about the model's limitations, potential biases in the data, or areas where additional feature engineering could be beneficial.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.

| Model         | HPO1 (time_limit) | HPO2 (presets) | HPO3 (hyperparameter_tune_kwargs) | Kaggle Score |
|---------------|-------------------|----------------|-----------------------------------|--------------|
| Initial       | 600               | best_quality   | N/A                               | 1.79080      |
| add_features  | 600               | best_quality   | N/A                               | 0.49546      |
| hpo           | 600               | best_quality   | {'scheduler':'local','searcher':'bayes'} | 0.52306   |




### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](img/model_test_score.png)

## Summary

In this project, we developed a predictive model for bike rentals using the Bike Sharing Demand dataset from Kaggle. This is a regression problem where the objective is to predict the number of bikes that will be rented at a given hour, based on features such as weather conditions, date and time.

We began with a raw dataset which we split into training and test data. An initial model was trained using AutoGluon's TabularPredictor with 'root_mean_squared_error' as the evaluation metric, and a time limit of 600 seconds. This initial model yielded a Kaggle score of 1.79080, indicating substantial room for improvement.

In order to enhance our model's predictive capabilities, we conducted an exploratory data analysis (EDA) and feature engineering. We extracted additional features from the 'datetime' column, including 'year', 'month', 'day', 'hour', and 'dayofweek'. Additionally, interaction features and categorized features were created based on the 'workingday', 'temp', 'atemp', 'windspeed' and 'humidity' columns. The newly added features significantly improved the model performance, yielding a reduced Kaggle score of 0.49546.

Further, we experimented with hyperparameter optimization. The optimization strategy employed involved the use of Bayesian optimization over hyperparameters, with a local scheduler for managing the computational resources during the search process. Surprisingly, this resulted in a slight decrease in performance, with a Kaggle score of 0.52306. This suggests that while hyperparameter tuning can be beneficial, it does not always guarantee improved results, and the initial model with added features provided the best performance in this case.

In conclusion, our feature engineering strategies significantly improved the predictive performance of our model. Despite the slight decrease in performance with hyperparameter optimization, the process provided valuable insights into the intricacies of model tuning. In future work, we might consider exploring additional feature engineering techniques, more sophisticated hyperparameter optimization strategies, or other ML models.
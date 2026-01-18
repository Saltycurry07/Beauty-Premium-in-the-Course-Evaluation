# Good Looking, Good Rating? Beauty Premium in the Course Evaluation

Authors: Kuang Sheng & Liyan Wang

  Do physically attractive professors receive more favorable student evaluations? 
  
  This project builds a machine-learning pipeline that predicts professors’ course evaluation outcomes from facial images. We will construct a biographical dataset of professors from several U.S. universities, combining portrait photos with third-party evaluation measures from RateMyProfessors (RMP).Using an established deep-learning model (Liang et al., 2018) for facial attractiveness prediction, we will generate a standardized “beauty score” for each professor and link it to their RMP ratings. 
  We will then train and compare multiple predictive models, such as linear regression with regularization (Ridge/Lasso), logistic regression (for high- vs. low-rating classification), and gradient-boosted trees (e.g., XGBoost), to assess predictive performance and identify the most suitable approach for this task.

# Division of Responsibilities

Kuang Sheng:

I carried out this part of the project by first using web scraping to collect faculty photos from university and departmental websites. I then trained logistic regression and gradient-boosted tree models for prediction, followed by a systematic evaluation of model performance. Finally, I compiled the results into a written report and visualized key findings to clearly communicate the outcomes of the analysis.

Liyan Wang:

I conducted this project by first using web scraping to collect teaching evaluation data from Rate My Professor. I then applied a deep learning model to estimate professors’ attractiveness levels based on profile photos obtained from official university websites. Using these features, I trained linear regression models to predict professors’ ratings and examine the relationship between perceived attractiveness and teaching evaluations. Finally, I compiled the results into a written report and visualized key findings to clearly present the main insights of the analysis.

## RateMyProfessors Crawler

This repository includes a crawler script that fetches professor ratings for UCLA and NYU (Computer Science and Fine Arts) from RateMyProfessors and writes them to a CSV file.

## Data of Rate My Professor
Please refer to the latest version of data through this link: 

## Faculty Photo Data

Please refer to the latest version of data through this link: https://drive.google.com/drive/folders/1rZJVfmevApVX-XWipRbNk7OwWT1ggDXz?usp=sharing

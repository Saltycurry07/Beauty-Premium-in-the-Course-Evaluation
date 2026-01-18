# Good Looking, Good Rating? Beauty Premium in the Course Evaluation

### Authors: Kuang Sheng & Liyan Wang

### Instructor: Markus Neumann

  Do physically attractive professors receive more favorable student evaluations? 
  
  This project builds a machine-learning pipeline that predicts professors’ course evaluation outcomes from facial images. We will construct a biographical dataset of professors from several U.S. universities, combining portrait photos with third-party evaluation measures from RateMyProfessors (RMP). Using an established deep-learning model (Liang et al., 2018) for facial attractiveness prediction, we will generate a standardized “beauty score” for each professor and link it to their RMP ratings. 
  We will then train and compare multiple predictive models, such as linear regression with regularization (Ridge/Lasso), logistic regression (for high- vs. low-rating classification), and gradient-boosted trees (e.g., XGBoost), to assess predictive performance and identify the most suitable approach for this task.

## Division of Responsibilities

### Kuang Sheng
- **Primary responsibilities**
  - Web-scrape faculty profile photos from university/department websites
  - Train predictive models (logistic regression; gradient-boosted trees such as XGBoost)
  - Evaluate model performance (e.g., MAE/RMSE/AUC; cross-validation as needed)
  - Draft the report and visualize key results
- **Tasks completed this week**
  - Proposed the research question and identified feasible data sources, scraping tools, and a facial beauty prediction model
  - Collected faculty photos from the NYU and UCLA computer science department websites

### Liyan Wang
- **Primary responsibilities**
  - Web-scrape professor ratings from RateMyProfessors (RMP)
  - Apply the pre-trained deep-learning model to generate professors’ beauty scores
  - Train baseline predictive models (linear regression with/without regularization)
  - Draft the report and visualize key results
- **Tasks completed this week**
  - Collected RMP data for NYU and UCLA to validate the feasibility of the data pipeline
  - Produced descriptive-statistics plots to summarize the pilot dataset


## RateMyProfessors Crawler

This repository includes a crawler script that fetches professor ratings for UCLA and NYU (Computer Science and Fine Arts) from RateMyProfessors and writes them to a CSV file.

## Data of Rate My Professor
Please refer to the latest version of data through this link: https://github.com/Saltycurry07/Beauty-Premium-in-the-Course-Evaluation/blob/main/rmp_ucla_nyu_professors.csv

## Faculty Photo Data

Please refer to the latest version of data through this link: https://drive.google.com/drive/folders/1rZJVfmevApVX-XWipRbNk7OwWT1ggDXz?usp=sharing

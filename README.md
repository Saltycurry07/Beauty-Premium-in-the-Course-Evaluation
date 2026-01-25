# Good Looking, Good Rating? Beauty Premium in the Course Evaluation

- Authors: Kuang Sheng & Liyan Wang

- Instructor: Markus Neumann

## Research Question

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

## Faculty Photo Scraper

This repository also includes a web scraper script that fetches professors' photos for UCLA and NYU (Computer Science) from university websites.

## Data of RateMyProfessors
Please refer to the latest version of data through this link: https://github.com/Saltycurry07/Beauty-Premium-in-the-Course-Evaluation/blob/main/rmp_ucla_nyu_professors.csv

## Data of Faculty Photo

Please refer to the latest version of data through this link: https://drive.google.com/drive/folders/1rZJVfmevApVX-XWipRbNk7OwWT1ggDXz?usp=sharing

## SCUT-FBP5500 Beauty Score Inference (CSV)

This repo includes a script that loads the pretrained SCUT-FBP5500 PyTorch models and appends a `beauty_score` column to a CSV containing image paths.

### 1) Download the pretrained model

From the official SCUT-FBP5500 release: https://github.com/HCIILAB/SCUT-FBP5500-Database-Release

- Download the **PyTorch** trained models archive and extract it locally.
- Choose one of the `.pth` files, such as `alexnet.pth` or `resnet18.pth`.

### 2) Prepare your CSV

Your CSV should include a column with image paths. For example:

```csv
name,image_path
Professor A,/path/to/image_a.jpg
Professor B,/path/to/image_b.jpg
```

### 3) Run inference

```bash
python beauty_score_from_csv.py \
  --input-csv professors.csv \
  --image-column image_path \
  --model-arch alexnet \
  --weights /path/to/alexnet.pth \
  --output-csv professors_with_scores.csv
```

If images are relative paths, they will be resolved relative to the CSV file location. Any failures are logged to an `_errors.txt` file alongside the output CSV.

## Baseline Models Performance
On the cleaned pilot sample (n = 38), Linear Regression and Ridge achieve MAE = 0.83 (RMSE = 0.99), a modest improvement over the mean-prediction baseline (MAE = 0.87; RMSE = 1.03). Ridge performs slightly better than OLS, which is consistent with regularization helping stabilize estimates in a small-sample setting. However, cross-validated R² remains negative, suggesting that the current inputs, primarily the raw beauty score plus categorical controls (school and department), capture limited out-of-sample variation in RMP ratings. The gradient-boosted tree baseline (HistGBR) does not outperform the mean baseline, indicating little evidence of robust non-linear patterns given the present features and sample size. In the coming weeks, we will expand the dataset across more universities and incorporate additional signals (e.g., number of reviews) to reduce noise and improve predictive performance.
![CV results](model/evaluation/MAE_mean.png)
![CV results](model/evaluation/RMSE_mean.png)
![CV results](model/evaluation/R2_mean.png)



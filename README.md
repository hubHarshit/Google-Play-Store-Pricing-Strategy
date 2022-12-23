# Google Play Store Pricing Strategy using Hypothesis Testing


# Context:
The biggest repository for data is available on the web, and therefore data scraping and analysis has become an unavoidable and integral part of every data scientist’s life. With digitization prevalent in everyone’s lives, apps have been a significant contributor to it. Amongst all the app stores available, Google Play store holds a global market share of 75 percent with gaming apps being the most popular. With most of the apps on Google Play Store being free, the revenue models for them is very uncertain with some relying on in-purchase ads and some having in-app purchases. Therefore, to measure an app's success, the focus is on the number of downloads instead of revenue along with its ratings and number of reviews.
Our analysis is focused on these attributes for an app developer to determine which metrics to focus on while creating an app that can lead to its success in terms of both downloads and revenue generation.


# Business Problem:
Our business problem pertains to a company planning to launch a gaming app, to give recommendations on price of the app, the availability of in-app ads in the app, and in-app purchases in the app. This analysis is done using exploratory data analysis of the attributes, the correlation of above mentioned attributes with the number of downloads and hypothesis formulation.

# Parameters measured for an app's success:
1. Number of downloads per day
2. Number of reviews per day
3. Total number of Ratings per day -  indication of popularity
4. Average rating – indication of likeability

# Data retrieval
Used Google's Play Store website https://play.google.com/store and scraped the data from there to obtain a dataset with several attributes including but not limited to reviews, score, rating, etc. The Python libraries used for these are Json, Selenium, Parsel, BeautifulSoup, Matplotlib, etc.
After scraping, we identified a lot of errors with the data. There was a lot of duplication, some columns were irrelavant to our analysis, some barely had any values and few had values like screenshots and urls which was of no use. So, we used data cleaning to remove duplicates, columns with less than 6 values and irrelevant columns. Once the data cleaning part was done, we used Exploratory data analysis to identify the characteristics of each significant attribute.

# Important attributes identified in the dataset:
1. Real installs – Total number of installs the app has, signifying total users
2. Score – Rating of the app on a scale of 0-5
3. Ratings – Total number of ratings given by users
4. Reviews – Total number of reviews available for the app
5. Price – Price of the app, ranging from 0 to 7.49 (both in $ and £)
6. Free – Indication of whether the app is free or paid
7. In-App product price – Price at which products within the app are available, ranges from 0 to $99.99 per item
8. Genre – Subcategory to which the gaming app belongs (Puzzle, Arcade, etc)
9. Ad supported – Whether the app has ads or not
10. Content rating – Whether the app is age appropriate (Everyone, Teen, Adults, etc)

# EDA analysis
i) We created a distribution and a box plot to identify the average score along with the outliers for an app. The skewness of the score is primarily between 3.5 to 4.5, with a few outliers in range 0-3.3.
ii) Average app rating bar graph re-validates the point mentioned above.
iii) The bar graph for Number of downloads in different genres tells us the popularity of each genre, with Arcade being the most popular.
iv) We build 3 pairwise plots to determine the relationship between number of installs, number of reviews, price and score with respect to:
    -an app being free or not
    -an app having in-app purchases or not
    -an app being ad supported or not
v) We also compare the number of downloads per price of the apps, in-app purchases of the app and ad supported apps for different genres.
vi) The correlation heatmap gives us an overview of how each variable is correlated with each other. Some highlights are:
    -There is a strong correlation between real installs and number of ratings
    -Similarly, there is a strong correlation between real installs and reviews as well
    -Price is somewhat negatively correlated with ratings, reviews and score, that means when price is higher, score is somewhat lower, and number of ratings/reviews are also low which indicates its lower popularity
    -Score is slightly positively correlated with ratings and reviews which means higher number of ratings usually lead to a somewhat higher score

## Reviews on app
The next part of the project was scraping the data on app reviews. The purpose of this is to identify the sentiment polarity and sentiment subjectivity in terms of the reviews obtained.
 Important attributes identified in the dataset:
1. Score – Rating of the app on a scale of 1-5
2. Content – Provides the review of the user on that particular app
3. Thumbsup Count – Amount the review was liked by other users
4. Subjectivity - This indicates the likeness of a user's review to the general review of the public. A higher subjectivity indicates the user's reviews are very much in line with the general public. This is between (0,1).
5. Polarity - A negative value means the review has a negative sentiment, positive value means review has a positive sentiment. This is between (-1,1).

# 
Distribution of subjectivity:
Subjectivity score distribution shows that the major reviews either have 0 value or 0.5 indicating people usually prefer giving general opinion rather than personal opinion according to their experience.

Sentiment analysis:
We use the Textblob library to generate sentiment ratings for each review for an app which was scraped and stored in the reviews.csv. We see that the polarity and subjectivity have a slight correlation.

# Genre taken: Arcade

# Hypothesis formulation
Free Apps:

H0: Average no. of downloads for free apps having Ads is greater than Average no. of downloads for free apps having in-app purchases.
Ha: Average no. of downloads for free apps having Ads is not greater than Average no. of downloads for free apps having in-app purchases.
One tail T-test result: 
We are not rejecting the null hypothesis as the p-value is greater than 0.05. Therefore, Average no. of downloads for free apps having app contains Ad is not greater than in-app purchases.

Paid Apps:
H0: Average no. of downloads for paid apps having in-app purchases is greater than Average no. of downloads for free apps having Ads.
Ha: Average no. of downloads for paid apps having in-app purchases is not greater than daily no. of downloads for free apps having Ads.
One tail T-test result:
We are not rejecting the null hypothesis as the p-value is greater than 0.05. Average no. of downloads for paid apps having in-app purchases is not greater than app contains Ad.

# Linear Regression

After getting results from the hypothesis, we created a linear regression for number of installs based on price, app being free or not, in app purchases, ad-supported, average review, number of days since release, polarity and subjectivity. From the regression above, we can observe that all the P-values are negligible hence this all the terms are significant.

Some notes:
[1] R² is computed without centering (uncentered) since the model does not contain a constant.
[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[3] The condition number is large, 2.22e+04. This might indicate that there are
strong multicollinearity or other numerical problems.

This model, with further tuning, can be used to predict number of downloads. Fine tuning can be done by hyperparameter tuning and scraping more data. This can be an excellent model for future number of download predictions.

Therefore, for a company to launch an app, it needs to focus on price, app being free or not, in app purchases, ad-supported, average review, number of days since release, polarity and subjectivity.

# Conclusion:
For a company to launch an app in ‘Arcade’ genre, the best revenue strategy to boost the number of downloads for a free app is to keep Ads.

For a company to launch an app in ‘Arcade’ genre, the best revenue strategy to boost the number of downloads for paid app is to keep in-app purchases.

For a company to launch an app, the following significant parameters of an app being free, in app purchases, ad-supported, polarity and subjectivity can give a good estimate of the number of downloads.


# Python Libraries used
import pandas as pd
import random
from bs4 import BeautifulSoup
import requests, lxml, re, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly 
from google_play_scraper import app
import json
from tqdm import tqdm
from pygments import highlight
pip install TextBlob
from textblob import TextBlob
from sklearn import preprocessing, svm
!pip install statsmodels




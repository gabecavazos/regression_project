# regression_project

# Overview:
This project aims to identify the drivers of home value and build a model that predicts the value of single family properties. The dataset used is Zillow data from 2017, representing properties in LA, Orange, and Ventura County California. The features of the dataset include bedrooms, bathrooms, area, year built, and FIPS code. The project suggests creating a new feature that combines bedrooms and bathrooms into one metric to improve the correlation of the value. Additionally, encoding FIPS and building separate models for each county can help improve the performance of the model.

# Data:
The dataset used in this project is Zillow data from 2017. It contains 2,142,803 rows representing single family properties in LA, Orange, and Ventura County California. The features of the dataset include bedrooms, bathrooms, area, year built, and FIPS code.

# Model:
The project aims to build a model that predicts the value of properties based on the features. The initial thoughts suggest that square footage (area) and age will have the biggest influence on tax_value. The final model chosen was a simple linear regression.

# Recommendations
The project recommends creating a new feature that combines bedrooms and bathrooms into one metric to improve the correlation of the value. Additionally, encoding FIPS and building separate models for each county can help improve the performance of the model.

# Conclusion:
The project concludes that tax value is at least partially dependent on all of the features selected. Finding a new metric to improve the correlation will improve the model. The current order of importance is area, bathrooms, and age.

# replicate process:
1. clone the repository
2. move your env file to the same folder
3. run report.ipynb sequentially from top to bottom
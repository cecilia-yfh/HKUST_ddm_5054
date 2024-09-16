#!/usr/bin/env python
# coding: utf-8

# # Problem 1: Investigation of Life Expectancy (100 points)

# # Data preprocessing

# <div class="alert alert-block alert-success"><b>Step 1</b>: Import data</div>

# In[1]:


# import data from Dataset: Life Expectancy Data.csv


# In[2]:


import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


# In[3]:


data = pd.read_csv('Life Expectancy Data.csv')


# <div class="alert alert-block alert-success"><b>Step 2</b>: Analysis</div>

# In[4]:


# Check the data 
data


# In[5]:


# View descriptive statistical analysis
data_stats = data.describe()
styled_stats = data_stats.style.set_table_styles([{'selector': 'th','props': [('background-color', 'lightgray')]}])
styled_stats


# In[6]:


column_names = data.columns
print(column_names)


# In[7]:


data = data.rename(columns=lambda x: x.strip())


# In[8]:


column_names = data.columns
print(column_names)


# <div class="alert alert-block alert-info"> 
# Missing Value Analysis
# </div>

# In[9]:


df_filtered = data[data.isnull().any(axis=1)]
df_filtered.head()


# In[10]:


# Check the number and percentage of NAN
missing_count = data.isnull().sum()
missing_percentage = data.isnull().mean() * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing Percentage': missing_percentage
})

print(missing_df)

# Plot number of missing values bar chart and proportional line chart
plt.figure(figsize=(10, 6))

ax1 = sns.barplot(x=missing_df.index,y='Missing Count',data=missing_df)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax1.set_xlabel('Variables')
ax1.set_ylabel('Missing Count')
ax1.set_title('Missing Values Count')

ax2 = ax1.twinx()
ax2.plot(missing_df.index,missing_df['Missing Percentage'],color='blue',marker='o')
ax2.set_ylabel('Missing Percentage', color='blue')
ax2.tick_params(axis='y',labelcolor='blue')
for x, y in zip(missing_df.index, missing_df['Missing Percentage']):
    ax2.annotate(f'{y:.2f}%', (x, y), textcoords='offset points', xytext=(0, 10), ha='center')

plt.show()


# In[11]:


# View the distribution of missing values
plt.figure(figsize=(10, 8))
sns.heatmap(data.isnull(), cmap='viridis', cbar=False)
plt.title('Missing Values Heatmap')
plt.show()


# <div class="alert alert-block alert-info"> 
# Missing Value handling
#     
# #### According to the data :Some rows can be deleted:(the percentage of NaN <5%):
# > **Life expectancy** : num:10, percentage:0.340368%  
# > **Adult Mortality** : num:10, percentage:0.340368%  
# > **BMI** : num:34, percentage:1.157250%  
# > **polio**: num:19, percentage:0.646698%  
# > **Diphtheria** num:19, percentage:0.646698%  
# > **thinness 1-19 years**  : num:34, percentage:1.157250%   
# > **thinness 5-9 years** : num:34, percentage:1.157250%
#     
# after delete the rows, the data name is "data_filtered"
# </div>

# In[12]:


columns_with_missing_values = ['Life expectancy', 'Adult Mortality', 'BMI', 'Polio', 'Diphtheria', 'thinness  1-19 years', 'thinness 5-9 years']
data_filtered = data.dropna(subset=columns_with_missing_values)


# In[13]:


data_filtered 


# <div class="alert alert-block alert-info"> 
# Missing Value handling
#     
# #### According to the data : Missing value filling
# 
# >**Alchohol & Total expenditure** :Missing values are evenly distributed and missing are consistent  
# >Alcohol                                    194            6.603131  
# >Total expenditure                          226            7.692308  
# 
# >**Hepatitis B** :There are many missing values and the distribution of missing values is relatively even.  
# >Hepatitis B                                553           18.822328    
# 
# >**GDP & Population & Income composition of resources & Schooling**   
# >The first two have many missing values, and missing value aggregation occurs. The latter two have missing values, and missing value aggregation occurs. Moreover, the missing rows of the missing values of the first two can be seen from the heat map that they cover the following The case of missing values in both  
# >GDP                                        448           15.248468  
# >Population                                 652           22.191967  
# >Income composition of resources            167            5.684139  
# >Schooling                                  163            5.547992
# 
#     
# </div>

# In[14]:


msno.matrix(data_filtered)


# In[15]:


plt.figure(figsize=(10, 8))
sns.heatmap(data_filtered.isnull(), cmap='viridis', cbar=False)
plt.title('Missing Values Heatmap')
plt.show()


# In[16]:


correlation_matrix = data_filtered.corr(method='pearson', numeric_only=True)
plt.figure(figsize=(18, 14))
#Set the upper triangle of the matrix to zero and only display the lower triangle.
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
cmap = sns.diverging_palette(250, 10, as_cmap=True)
sns.heatmap(correlation_matrix, annot=True, cmap=cmap, mask=mask, vmin=-1, vmax=1)
plt.tight_layout()
plt.show()


# In[17]:


missing_corr = data_filtered.isnull().corr()
plt.figure(figsize=(18, 14))
sns.heatmap(missing_corr, annot=True, cmap='coolwarm')


# <div class="alert alert-block alert-info"> 
# Missing Value handling
#     
# #### What's more, According to the heatmap  
# Three groups need consider multi-column padding
# > **Alcohol** and **Total expenditure**  
# > **Population** and **GDP**  
# > **Income composition of resouces** and**Schooling**  
# 
# Single column filling
# > **Hepatitis B** ； 
# 
# If I have more time, I will try multi-column filling such as knn/mice/machine learning and find the most suitable one. But at here, I just use single column filling.
# </div>

# In[19]:


#filling Hepatitis B'Nan
#计算每个字段的平均值
mean_alcohol = data_filtered.loc[:, 'Alcohol'].mean()
mean_total_expenditure = data_filtered.loc[:, 'Total expenditure'].mean()
mean_population = data_filtered.loc[:, 'Population'].mean()
mean_gdp = data_filtered.loc[:, 'GDP'].mean()
mean_income_composition = data_filtered.loc[:, 'Income composition of resources'].mean()
mean_schooling = data_filtered.loc[:, 'Schooling'].mean()
mean_hepatitis_b = data_filtered.loc[:, 'Hepatitis B'].mean()

#填充缺失值

data_filtered.loc[:, 'Alcohol'] = data_filtered.loc[:, 'Alcohol'].fillna(mean_alcohol)
data_filtered.loc[:, 'Total expenditure'] = data_filtered.loc[:, 'Total expenditure'].fillna(mean_total_expenditure)
data_filtered.loc[:, 'Population'] = data_filtered.loc[:, 'Population'].fillna(mean_population)
data_filtered.loc[:, 'GDP'] = data_filtered.loc[:, 'GDP'].fillna(mean_gdp)
data_filtered.loc[:, 'Income composition of resources'] = data_filtered.loc[:, 'Income composition of resources'].fillna(mean_income_composition)
data_filtered.loc[:, 'Schooling'] = data_filtered.loc[:, 'Schooling'].fillna(mean_schooling)
data_filtered.loc[:, 'Hepatitis B'] = data_filtered.loc[:, 'Hepatitis B'].fillna(mean_hepatitis_b)

import warnings
warnings.filterwarnings('ignore')


# In[20]:


# filling Alco


# In[21]:


data_stats = data_filtered.describe()
styled_stats = data_stats.style.set_table_styles([{'selector': 'th','props': [('background-color', 'lightgray')]}])
styled_stats


# <div class="alert alert-block alert-info"> 
# Coding
# </div>

# In[22]:


data_filtered.loc[:,'Status'] = data_filtered['Status'].replace({'Developed': 1, 'Developing': 0})

import warnings
warnings.filterwarnings('ignore')


# In[23]:


data_filtered


# <div class="alert alert-block alert-info"> 
# Outliers
# </div>

# In[24]:


plt.figure(figsize=(30, 10))
data_filtered.boxplot()
plt.show()


# ## 1) Report the summary of the linear model. 
# What are the predicting variables actually affecting the life expectancy? Justify your answer based on the outputs of linear regression model

# <div class="alert alert-block alert-info"> 
# According to Coef and P-value:  
#     
# **When Coef is not too small and P-value<0.5, the variable is valid**   
# 
# > Status;   Adult Mortality;   infant deaths;   Alcohol;   Hepatitis B;   BMI;   under-five deaths;   Polio;   Total expenditure;   Diphtheria;   thinness  1-19 years;   Income composition of resources;   Schooling.
# 
# </div>

# In[25]:


# Extract the independent variable X and dependent variable y
X = data_filtered.drop(columns=['Life expectancy', 'Country'])
y = data_filtered['Life expectancy']

# Create a new dataframe
data_regression = pd.concat([X, y], axis=1)


# In[26]:


import statsmodels.api as sm
X = sm.add_constant(X)
regression_model = sm.OLS(y, X)
regression_results = regression_model.fit()
summary = regression_results.summary()
print(summary)


# ## 2) Construct the 95% confidence intervals 
# for the coefficient of “Adult Mortality” and “HIV/AIDS”. Are you confident that these predictors have negative impact on the life expectancy? Explain why.

# <div class="alert alert-block alert-info"> 
# 
# **YES**
# These predictors have negative impact on expectancy.  
# For the reason that these predictors' Lower_limit and upper_limit of the confidence interval are both less than zero. So these independent variables have negative impact and the impact is statistically significant.
# 
# </div>
# 
# 

# In[41]:


coef_table = summary.tables[1]  

coef_data = coef_table.data[1:]  
coef_names = [row[0] for row in coef_data]  
coef_values = [float(row[1]) for row in coef_data] #coefname&value

conf_int_data = coef_table.data[1:]  
conf_int_values = [(float(row[5]), float(row[6])) for row in conf_int_data]  # interval

# find the index of “Adult Mortality”“HIV/AIDS"
adult_mortality_index = coef_names.index("Adult Mortality")
hiv_aids_index = coef_names.index("HIV/AIDS")

#Delivery the coef and confidence interval
coef_adult_mortality = coef_values[adult_mortality_index]
conf_int_adult_mortality = conf_int_values[adult_mortality_index]
coef_hiv_aids = coef_values[hiv_aids_index]
conf_int_hiv_aids = conf_int_values[hiv_aids_index]

print("coef,95% Confidence Interval for Adult Mortality Coefficient:", (coef_adult_mortality, conf_int_adult_mortality))
print("coef,95% Confidence Interval for HIV/AIDS Coefficient:", (coef_hiv_aids,conf_int_hiv_aids))


# ## 3) Construct the 97% confidence intervals 
# for the coefficient of “Schooling” and “Alcohol”. Explain how these predictors impact the life expectancy

# <div class="alert alert-block alert-info"> 
# 
# **97% Confidence Interval for the Coefficient of Schooling: (0.63, 0.824)**  
# **97% Confidence Interval for the Coefficient of Alcohol: (0.003, 0.117)**  
# these predictors have positive impact on the life expectancy. (Because their conficence intervals lower_limit and upper_limit are all above zero.
# </div>
# 
# 

# In[35]:


# Calculate t
from scipy.stats import t

# degrees of freedom
N = 2888
df = N - 2  
# 97% confidence
confidence_level = 0.97
t = t.ppf(1 - (1 - confidence_level) / 2, df)
t


# In[45]:


import statsmodels.api as sm

coef_table = summary.tables[1]  
coef_data = coef_table.data[1:]  
coef_names = [row[0] for row in coef_data]  

# Find the index positions of "Schooling" and "Alcohol"
schooling_index = coef_names.index("Schooling")
alcohol_index = coef_names.index("Alcohol")

# Extract the coefficient values
coef_schooling = regression_results.params[schooling_index]
coef_alcohol = regression_results.params[alcohol_index]

# Calculate the standard errors
se_schooling = regression_results.bse[schooling_index]
se_alcohol = regression_results.bse[alcohol_index]

# Use the t-value before 
# Calculate the margin of error
me_schooling = t * se_schooling
me_alcohol = t * se_alcohol

# Calculate the confidence intervals
ci_schooling = (round(coef_schooling - me_schooling,3), round(coef_schooling + me_schooling,3))
ci_alcohol = (round(coef_alcohol - me_alcohol,3), round(coef_alcohol + me_alcohol,3))

# Print the results
print("97% Confidence Interval for the Coefficient of Schooling:", ci_schooling)
print("97% Confidence Interval for the Coefficient of Alcohol:", ci_alcohol)


# ## 4) The top-seven infulential predictors.
# Based on the p-values, which are the top-seven most influential predictors? Use these predictors to fit a smaller model and report the summary.

# In[91]:


p_values = regression_results.pvalues[1:]# p without considering the constant term
predictors_top_7 = p_values.sort_values().index[:7]

for pre in predictors_top_7:
    p_value = p_values[pre]
    print(f"{pre}: {p_value}")


# In[93]:


# create a new model
X2 = sm.add_constant(data_filtered[predictors_top_7])
# Extract the independent variable X and dependent variable y
y = data_filtered['Life expectancy']
model = sm.OLS(y, X2)
new_model_results = model.fit()
summary2 = new_model_results.summary()
print(summary2)


# ## 5) Predict life expectancy
# Use the smaller model to predict the life expectancy if a new observation is given with Year=2008, Status=Developed, Adult Mortality=125, infant deaths=94, Alcohol=4.1, percentage expenditure=100, Hepatitis B=20, Measles=13, BMI=55, under-five deaths=2, Polio=12, Total expenditure=5.9, Diphtheria=12,
# HIV/AIDS=0.5, GDP=5892,Population=1.34 × 106, Income composition of resources=0.9, Schooling=18.
# 
# Report the 99% confidence interval for your prediction.

# <div class="alert alert-block alert-info"> 
# 
# **99% Confidence Interval:[83.46962031,83.4891509]**  
# Lower Bound: [83.46962031]
# Upper Bound: [83.4891509]
# 
# </div>
# 

# In[95]:


X2.head()


# In[97]:


# Create the new observation
new_observation = [1,0.5,125,18,2,94,0.9,12]
# Predict the life expectancy 
predicted_life_expectancy = new_model_results.predict(new_observation)
# Calculate the standard error of the prediction
prediction_standard_error = new_model_results.bse[-1]  # Assuming the last coefficient corresponds to the dependent variable

import scipy.stats
# Calculate the critical value for a 99% confidence level
critical_value = scipy.stats.t.ppf(0.995, new_model_results.df_resid)

# Calculate the margin of error
margin_of_error = critical_value * prediction_standard_error

# Calculate the lower and upper bounds of the confidence interval
lower_bound = predicted_life_expectancy - margin_of_error
upper_bound = predicted_life_expectancy + margin_of_error

print("Predicted Life Expectancy:", predicted_life_expectancy)
print("99% Confidence Interval:")
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)


# ## 6) Use AIC to compare the full model and the smaller model

# <div class="alert alert-block alert-info"> 
# Full model has a lower AIC
# </div>

# In[98]:


# Calculate the AIC values
aic_full = regression_results.aic
aic_small = new_model_results.aic

# Compare the AIC values
if aic_full < aic_small:
    print("Full model has a lower AIC, indicating a better fit.")
elif aic_small < aic_full:
    print("Smaller model has a lower AIC, indicating a better fit.")
else:
    print("Both models have the same AIC.")


# In[ ]:





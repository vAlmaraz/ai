import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Read data
heartData = pd.read_csv("../datasets/heart.data.csv")
# Check data
print(heartData.describe())
# Check for independence of observations (aka no autocorrelation)
# The correlation of 1.5% is small enough that we can include both variables in the model
print(heartData['biking'].corr(heartData['smoking']))
# Check for normality of dependent variable (heart disease) with a histogram
# The distribution of observations is roughly bell shaped so we can proceed
plt.hist(heartData['heart.disease'])
plt.show()
# Check for linearity between heart disease and biking, and disease and smoking
# Both look roughly linear
plt.scatter(heartData['biking'], heartData['heart.disease'])
plt.xlabel('Biking')
plt.ylabel('Heart Disease')
plt.show()
plt.scatter(heartData['smoking'], heartData['heart.disease'])
plt.xlabel('Smoking')
plt.ylabel('Heart Disease')
plt.show()

# Train the model
X = heartData[['biking', 'smoking']]
y = heartData['heart.disease']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Check for homoscedasticity
# The residuals vs. fitted plot can be used to check for heteroscedasticity.
# If the red lines representing the mean of the residuals are roughly horizontal and centered around zero,
# it indicates no outliers or biases that would make a linear regression invalid.
plt.scatter(model.fittedvalues, model.resid)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()

# Draw it
biking_values = np.linspace(heartData['biking'].min(), heartData['biking'].max(), num=30)
smoking_values = [heartData['smoking'].min(), heartData['smoking'].mean(), heartData['smoking'].max()]
plotting_data = pd.DataFrame({'biking': biking_values})
plotting_data['smoking'] = np.repeat(smoking_values, len(biking_values) // len(smoking_values))
plotting_data['predicted.y'] = model.predict(sm.add_constant(plotting_data[['biking', 'smoking']]))
plotting_data['smoking'] = round(plotting_data['smoking'], 2)
plotting_data['smoking'] = plotting_data['smoking'].astype('category')

heart_plot = sns.scatterplot(data=heartData, x='biking', y='heart.disease')
heart_plot = sns.lineplot(data=plotting_data, x='biking', y='predicted.y', hue='smoking', linewidth=1.25)
plt.show()

heart_plot = sns.scatterplot(data=heartData, x='biking', y='heart.disease')
heart_plot = sns.lineplot(data=plotting_data, x='biking', y='predicted.y', hue='smoking', linewidth=1.25)
heart_plot.set(title='Rates of heart disease (% of population) \n as a function of biking to work and smoking',
               xlabel='Biking to work (% of population)',
               ylabel='Heart disease (% of population)')
plt.show()
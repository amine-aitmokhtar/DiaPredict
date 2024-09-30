import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file
import matplotlib.pyplot as plt  # to plot charts
import seaborn as sns  # used for data visualization
import warnings  # avoid warning flash

from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Loading the dataset¶
df = pd.read_csv("diabetes.csv")

# Exploratory Data Analysis
print(df.head())  # get familier with dataset, display the top 5 data records
print(df.shape)  # getting to know about rows and columns we're dealing with - 768 rows , 9 columns
print(df.columns)  # learning about the columns
print(df.dtypes)  # knowledge of data type helps for computation
print(
    df.info())  # Print a concise summary of a DataFrame. This method prints information about a DataFrame including the index dtype and columns, non-null values and memory usage.
print(df.describe())  # helps us to understand how data has been spread across the table.
# count :- the number of NoN-empty rows in a feature.
# mean :- mean value of that feature.
# std :- Standard Deviation Value of that feature.
# min :- minimum value of that feature.
# max :- maximum value of that feature.
# 25%, 50%, and 75% are the percentile/quartile of each features.

# ****** CONCLUSION ******
# We observe that min value of some columns is 0 which cannot be possible medically.Hence in the data cleaning process
# we'll have to replace them with median/mean value depending on the distribution. Also in the max column we can see insulin
# levels as high as 846! We have to treat outliers.

# Data Cleaning

# dropping duplicate values - checking if there are any duplicate rows and dropping if any
df = df.drop_duplicates()

print(df)

# check for missing values, count them and print the sum for every column
print(df.isnull().sum())  # conclusion :- there are no null values in this dataset

# checking for 0 values in 5 columns , Age & DiabetesPedigreeFunction do not have have minimum 0 value so no need to replace , also no. of pregnancies as 0 is possible as observed in df.describe
print(df[df['BloodPressure'] == 0].shape[0])
print(df[df['Glucose'] == 0].shape[0])
print(df[df['SkinThickness'] == 0].shape[0])
print(df[df['Insulin'] == 0].shape[0])
print(df[df['BMI'] == 0].shape[0])

# replacing 0 values with median of that column
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())  # normal distribution
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())  # normal distribution
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())  # skewed distribution
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())  # skewed distribution
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())  # skewed distribution

# Data Visualization

sns.countplot(x='Outcome', data=df)

# Then, create histograms for each feature
df.hist(bins=10, figsize=(10, 10))
plt.show()

plt.figure(figsize=(16, 12))
sns.set_style(style='whitegrid')
plt.subplot(3, 3, 1)
sns.boxplot(x='Glucose', data=df)
plt.subplot(3, 3, 2)
sns.boxplot(x='BloodPressure', data=df)
plt.subplot(3, 3, 3)
sns.boxplot(x='Insulin', data=df)
plt.subplot(3, 3, 4)
sns.boxplot(x='BMI', data=df)
plt.subplot(3, 3, 5)
sns.boxplot(x='Age', data=df)
plt.subplot(3, 3, 6)
sns.boxplot(x='SkinThickness', data=df)
plt.subplot(3, 3, 7)
sns.boxplot(x='Pregnancies', data=df)
plt.subplot(3, 3, 8)
sns.boxplot(x='DiabetesPedigreeFunction', data=df)

plt.show()

from pandas.plotting import scatter_matrix

scatter_matrix(df, figsize=(20, 20));
# we can come to various conclusion looking at these plots for example  if you observe 5th plot in pregnancies with insulin, you can conclude that women with higher number of pregnancies have lower insulin

# Feature Selection
corrmat = df.corr()
sns.heatmap(corrmat, annot=True)
plt.show()

# ****** CONCLUSION ******
# Observe the last row 'Outcome' and note its correlation scores with different features. We can observe that Glucose,
# BMI and Age are the most correlated with Outcome. BloodPressure, Insulin, DiabetesPedigreeFunction are the least correlated,
# hence they don't contribute much to the model so we can drop them

df_selected = df.drop(['BloodPressure', 'Insulin', 'DiabetesPedigreeFunction'], axis='columns')

# - Handling Outliers
# To identify Outliers we can use Box plots, Scatter plot, Z score

print(df)
from sklearn.preprocessing import QuantileTransformer

x = df_selected
quantile = QuantileTransformer()
X = quantile.fit_transform(x)
df_new = quantile.transform(X)
df_new = pd.DataFrame(X)
df_new.columns = ['Pregnancies', 'Glucose', 'SkinThickness', 'BMI', 'Age', 'Outcome']
print(df_new.head())

plt.figure(figsize=(16, 12))
sns.set_style(style='whitegrid')
plt.subplot(3, 3, 1)
sns.boxplot(x=df_new['Glucose'], data=df_new)
plt.subplot(3, 3, 2)
sns.boxplot(x=df_new['BMI'], data=df_new)
plt.subplot(3, 3, 3)
sns.boxplot(x=df_new['Pregnancies'], data=df_new)
plt.subplot(3, 3, 4)
sns.boxplot(x=df_new['Age'], data=df_new)
plt.subplot(3, 3, 5)
sns.boxplot(x=df_new['SkinThickness'], data=df_new)
plt.show()

# Split the Data Frame into X and y
target_name = 'Outcome'
y = df_new[target_name]  # given predictions - training data
X = df_new.drop(target_name, axis=1)  # dropping the Outcome column and keeping all other columns as X

print(X.head())  # contains only independent features
print(y.head())  # contains dependent feature

# TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)  # splitting data in 80% train, 20%test

X_train.shape, y_train.shape
X_test.shape, y_test.shape

# Classification Algorithms

# - KNN
# - SVM
# - Decision Tree
# - Random Forest
# - Logistic Regression

## K Nearest Neighbours :-

# KNN algorithm, is a non-parametric algorithm that classifies data points based on their proximity and association to other available data.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

# List Hyperparameters to tune
knn = KNeighborsClassifier()
n_neighbors = list(range(15, 25))
p = [1, 2]
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']

# convert to dictionary
hyperparameters = dict(n_neighbors=n_neighbors, p=p, weights=weights, metric=metric)

# Making model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=knn, param_grid=hyperparameters, n_jobs=-1, cv=cv, scoring='f1', error_score=0)

best_model = grid_search.fit(X_train, y_train)

# Best Hyperparameters Value
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

# Predict testing set
knn_pred = best_model.predict(X_test)

print("Classification Report is:\n", classification_report(y_test, knn_pred))
print("\n F1:\n", f1_score(y_test, knn_pred))
print("\n Precision score is:\n", precision_score(y_test, knn_pred))
print("\n Recall score is:\n", recall_score(y_test, knn_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test, knn_pred))
plt.show()

# Support Vector Machine :-

# It is typically leveraged for classification problems, constructing a hyperplane where the distance between two classes of data points
# is at its maximum. This hyperplane is known as the decision boundary, separating the classes of data points (e.g., has diabetes vs
# doesn't have diabetes ) on either side of the plane.

model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']

# define grid search
grid = dict(kernel=kernel, C=C, gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='f1', error_score=0)

grid_result = grid_search.fit(X, y)

svm_pred = grid_result.predict(X_test)

print("Classification Report is:\n", classification_report(y_test, svm_pred))
print("\n F1:\n", f1_score(y_test, knn_pred))
print("\n Precision score is:\n", precision_score(y_test, knn_pred))
print("\n Recall score is:\n", recall_score(y_test, knn_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test, svm_pred))
plt.show()

##Decision Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
dt = DecisionTreeClassifier(random_state=42)

# Create the parameter grid based on the results of random search
params = {
    'max_depth': [5, 10, 20,25],
    'min_samples_leaf': [10, 20, 50, 100,120],
    'criterion': ["gini", "entropy"]
}

grid_search = GridSearchCV(estimator=dt,
                           param_grid=params,
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")

best_model=grid_search.fit(X_train, y_train)
dt_pred=best_model.predict(X_test)


print("Classification Report is:\n",classification_report(y_test,dt_pred))
print("\n F1:\n",f1_score(y_test,dt_pred))
print("\n Precision score is:\n",precision_score(y_test,dt_pred))
print("\n Recall score is:\n",recall_score(y_test,dt_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,dt_pred))
plt.show()

## Random Forest :-

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# define models and parameters
model = RandomForestClassifier()
n_estimators = [1800]
max_features = ['sqrt', 'log2']

# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

best_model = grid_search.fit(X_train, y_train)

rf_pred=best_model.predict(X_test)

print("Classification Report is:\n",classification_report(y_test,rf_pred))
print("\n F1:\n",f1_score(y_test,knn_pred))
print("\n Precision score is:\n",precision_score(y_test,knn_pred))
print("\n Recall score is:\n",recall_score(y_test,knn_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,rf_pred))
plt.show()

## 9.6 Logistic Regression:-
# Logistical regression is selected when the dependent variable is categorical, meaning they have binary outputs, such as "true" and "false" or "yes" and "no."


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score

reg = LogisticRegression()
reg.fit(X_train,y_train)
lr_pred=reg.predict(X_test)

print("Classification Report is:\n",classification_report(y_test,lr_pred))
print("\n F1:\n",f1_score(y_test,lr_pred))
print("\n Precision score is:\n",precision_score(y_test,lr_pred))
print("\n Recall score is:\n",recall_score(y_test,lr_pred))
print("\n Confusion Matrix:\n")
sns.heatmap(confusion_matrix(y_test,lr_pred))

plt.show()

#************************* Final Conclusion *****************************************

# In this study of diabetes prediction, we evaluated several machine learning models, including K-Nearest Neighbors (KNN),
# Support Vector Machine (SVM), Logistic Regression, Decision Tree, and Random Forest.

# The results obtained are summarized below:

# For KNN and SVM, the F1 scores, precision, and recall are all approximately equal at around 0.67, indicating relatively similar performance of these two models in diabetes
# prediction. In contrast, logistic regression exhibits a slightly lower F1 score, with a precision of 0.69 and a recall of 0.57, suggesting a moderate ability to predict
# diabetes. The decision tree shows a lower F1 score of about 0.57, with high precision (0.78) but relatively low recall (0.45), implying a tendency to misclassify positive
# cases. Finally, the random forest shows results similar to KNN and SVM, with an F1 score, precision, and recall of approximately 0.67. In conclusion, our study demonstrates
# that KNN, SVM, and the random forest appear to be the most promising models for diabetes prediction in our dataset, with balanced F1 scores between precision and recall.
# However, it is essential to consider other factors such as generalization and interpretability when selecting the final model for clinical application. Future work could
# also explore additional optimization techniques to further enhance the performance of these models.
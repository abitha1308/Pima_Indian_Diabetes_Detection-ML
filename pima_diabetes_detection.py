# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 14:20:32 2021

@author: Abitha M
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

print("\n\t**** PIMA'S INDIAN DIABETES DATA ANALYSIS & DETECTION ****\n\n")
df = pd.read_csv("pima-indians-diabetes.csv")
print("\nFIRST FIVE RECORDS OF THE DATASET")
print(df.head(5))
df.shape
print("\nDESCRIPTION OF THE DATASET")
print(df.describe())

df.hist(bins=10,figsize=(12,12))
#plt.title("HISTOGRAM TO CHECK FOR NORMAL DISTRIBUTION")
plt.show()

# find normal distribution or not using SHAPIRO WILK TEST OF NULL HYPOTHESIS
from scipy.stats import shapiro


#glucose concentration
stat1,p1 = shapiro(df["Plasma_glucose_conc"])
print("\nShapiro test on Plasma glucose concentration")
print("\tStatistical value={:.3f} P value={:.3f}".format(stat1,p1))

#BP
stat2,p2 = shapiro(df["BP"])
print("\nShapiro test on BP")
print("\tStatistics={:.3f} P value={:.3f}".format(stat2,p2))

#BMI
stat3,p3 = shapiro(df["BMI"])
print("\nShapiro test on BMI")
print("\tStatistics={:.3f} P value={:.3f}".format(stat3,p3))

stat4,p4 = shapiro(df["Triceps_thickness"])
print("\nShapiro test on Triceps thickness of skin")
print("\tStatistics={:.3f} P value={:.3f}".format(stat4,p4))


sns.violinplot(y="Pregnancy_count",x="Class_variable",palette="viridis",split=True,data=df)
plt.title("Pregnancy count Vs Class variable")
plt.show()

sns.violinplot(y="Plasma_glucose_conc",x="Class_variable",palette="viridis",split=True,data=df)
plt.title("Plasma glucose concentration Vs Class variable before handling zero")
plt.show()
df1 = df.loc[df["Class_variable"]==1]

df2 = df.loc[df["Class_variable"]==0]

#replace 0 with median value in the plasma glucose column

df1 = df1.replace({"Plasma_glucose_conc":0},np.median(df1["Plasma_glucose_conc"]))
df2 = df2.replace({"Plasma_glucose_conc":0},np.median(df2["Plasma_glucose_conc"]))

df3 = [df1,df2]
df = pd.concat(df3)

sns.violinplot(y="Plasma_glucose_conc",x="Class_variable",palette="viridis",split=True,data=df)
plt.title("Plasma glucose concentration Vs Class variable after handling zero")
plt.show()


sns.violinplot(x="Class_variable",y="BP",split=True,palette="viridis",data=df)
plt.title("BP Vs Class variable before handling zero")
plt.show()


df1 = df.loc[df["Class_variable"] == 0]
df2 = df.loc[df["Class_variable"] == 1]

df1 = df1.replace({"BP":0},np.median(df1["BP"]))
df2 = df2.replace({"BP":0},np.median(df2["BP"]))

df = pd.concat([df1,df2])

sns.violinplot(x="Class_variable",y="BP",split=True,palette="viridis",data=df)
plt.title("BP Vs Class variable after handling zero")
plt.show()

sns.violinplot(x="Class_variable",y="BMI",split=True,palette="viridis",data=df)
plt.title("BMI Vs Class variable before handling zero")
plt.show()

#Replace 0 with median of Bmi

df1 = df.loc[df["Class_variable"] == 0]
df2 = df.loc[df["Class_variable"] == 1]

df1 = df1.replace({"BMI":0},np.median(df1["BMI"]))
df2 = df2.replace({"BMI":0},np.median(df2["BMI"]))

df = pd.concat([df1,df2])

sns.violinplot(x="Class_variable",y="BMI",split=True,palette="viridis",data=df)
plt.title("BMI Vs Class variable after handling zero")
plt.show()

sns.violinplot(x="Class_variable",y="Triceps_thickness",split=True,palette="viridis",data=df)
plt.title("Triceps thickness Vs Class variable before handling zero")
plt.show()

#Replace 0 with median of Bp
df1 = df.loc[df["Class_variable"] == 0]
df2 = df.loc[df["Class_variable"] == 1]
#print(df1.describe())
#print(df2.describe())
df1 = df1.replace({"Triceps_thickness":0},np.median(df1["Triceps_thickness"]))
df2 = df2.replace({"Triceps_thickness":0},np.median(df2["Triceps_thickness"]))

df = pd.concat([df1,df2])

sns.violinplot(x="Class_variable",y="Triceps_thickness",split=True,palette="viridis",data=df)
plt.title("Triceps thickness Vs Class variable after handling zero")
plt.show()

sns.violinplot(x="Class_variable",y="Serum_insulin",split=True,palette="viridis",data=df)
plt.title("Serum insulin Vs Class variable before handling zero")
plt.show()

#Replace 0 with median of Bp
df1 = df.loc[df["Class_variable"] == 0]
df2 = df.loc[df["Class_variable"] == 1]
#print(df1.describe())
#print(df2.describe())
df1 = df1.replace({"Serum_insulin":0},np.median(df1["Serum_insulin"]))
df2 = df2.replace({"Serum_insulin":0},np.median(df2["Serum_insulin"]))


df = pd.concat([df1,df2])

sns.violinplot(x="Class_variable",y="Serum_insulin",split=True,palette="viridis",data=df)
plt.title("Serum Insulin Vs Class variable after handling zero")
plt.show()

# no 0's found in pedigree function but diabetic shows higher pedigree function

sns.violinplot(x="Class_variable",y="Pedigree_function",data=df,palette="viridis",split=True)
plt.title("Pedigree function Vs Class variable")
plt.show()


df10=df.corr()
df10["Age"]

#heatmaps using corr()
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),cmap="bwr",vmax=1,vmin=-1,annot=True)
plt.title("Heatmap to represent correlation between independent and dependent variables")
plt.show()

#Co-efficient of correlation (r) if r>0.70 then multi collinearity exists
print("\nCo-efficient of correlation (r) for Class_variable: ")
print(df10["Class_variable"])

#SPLIT DATA TO USE IN ML ALGORITHMS
y = df.Class_variable
x = df.drop("Class_variable",axis=1)

col = x.columns


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(x)

data_x = pd.DataFrame(X,columns=col)


#Training & testing variables

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data_x,y,test_size=0.15,random_state=45)

#handling imbalance using over sampling technique

y_train.hist(bins=10,figsize=(12,12))
plt.title("Imbalance in the training data of the dependent variable")
plt.show()


from imblearn.over_sampling import SMOTE

smt = SMOTE()

x_train,y_train = smt.fit_sample(x_train,y_train)

# balanced training Y variable

y_train.hist(bins=10,figsize=(7,7))
plt.title("Balanced training Y variable after Over Sampling Technique")
plt.show()


#   MODEL FITTING


# Try with Random Forest classifier 
print("\nUsing Random Forest Classifier !!\n")

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix
l = RandomForestClassifier(n_estimators=300,bootstrap=True,max_features='sqrt')

l.fit(x_train,y_train)
score = l.score(x_test,y_test)
scoret = l.score(x_train,y_train)
scorep = l.score(x_test,l.predict(x_test))

print("\tAccuracy with training data  = {:.3f}".format(scoret))
print("\tAccuracy with testing data = {:.3f}".format(score))
print("\tAccuracy with the predicted testing data = {:.3f}".format(scorep))


y_pred = l.predict(x_test)
cm= confusion_matrix(y_test,y_pred)
print(cm)

plt.figure(figsize=(10,10))
sns.heatmap(cm,cmap="Greens",annot=True,linecolor="black",linewidth=1)
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.show()



#using x_test predict y values

print("\nPredicted Accuracy of the model fitted to the data\n")
print("\tAccuracy score = {}".format(accuracy_score(y_test,y_pred)))
print("\tF1 Score = {}".format(f1_score(y_test,y_pred)))
print("\tPrecision Score = {}".format(precision_score(y_test,y_pred)))
print("\tRecall Score = {}".format(recall_score(y_test,y_pred)))


# Due to the presence of outliers
#Random forest classifier model provides higher accuracy than other models

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))


print("\n\n")
print("<--- PIMA'S INDIAN DIABETES DETECTION --->\n\n")

pc = float(input("Enter pregnancy count: "))
gc = float(input("Enter glucose concentration in the plasma: "))
age = float(input("Enter your age: "))
bp = float(input("Enter your blood pressure: "))
tt = float(input("Enter the thickness of skin: "))
si = float(input("Enter serum insulin level: "))
bmi = float(input("Enter body mass index(bmi): "))
pf = float(input("Enter pedigree function : "))

userinput = [pc,gc,bp,tt,si,bmi,pf,age]


outcome = l.predict([userinput])

print("\n-----------------------------------------")
if (outcome[0] == 0):
    print("\n\tYour are not a diabetic person :)")
    
else:
    print("\nYou are a diabetic person :(")

print("\n-----------------------------------------")



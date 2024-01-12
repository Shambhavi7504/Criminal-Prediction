1 import pandas as pd 2 import numpy as np
3 import matplotlib.pyplot as plt
4 import seaborn as sns
df=pd.read_csv('train.csv')
print(df.head(5))
print(df.isna().sum())
df.describe()
def plot_dis(var):
    fig, ax=plt.subplots(nrows=1)
    sns.countplot(x=var,hue='Criminal',data=df,ax=ax)
    plt.show()
for i in df.columns[1:]:
    plot_dis(i)
df.dropna(inplace=True)
df['Criminal'].value_counts()

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix ,plot_roc_curve
from imblearn.over_sampling import SMOTE
smote = SMOTE()
clf=ExtraTreeClassifier()
clf.fit(X_re,y_re)
clf.score(x_test,y_test)
plot_roc_curve(clf,x_test,y_test)
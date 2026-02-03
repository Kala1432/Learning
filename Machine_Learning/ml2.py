import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data= fetch_california_housing(as_frame=True)
df=data.frame

print("Dataset Sample.")
print(df.head())

correlation_matrix=df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

def plot_heatmap(corr_matrix):
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix,annot=True,fmt=".2f",cmap="coolwarm",cbar=True,square=True,linewidth=0.5)
    plt.title("correlation Matrix Heatmap",fontsize=16)
    plt.show()

def plot_pairplot(df):
    sns.pairplot(df,diag_kind="kde",corner=True,plot_kws={'alpha':0.5},diag_kws={'fill':True})
    plt.suptitle("pair plot if Numerical Features",y=1.02,fontsize=16)
    plt.show()
plot_heatmap(correlation_matrix)
plot_pairplot(df)


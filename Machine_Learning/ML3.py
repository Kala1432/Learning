import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris=load_iris()
X=iris.data
y=iris.target
target_names=iris.target_names
scaler=StandardScaler()
x_standardized=scaler.fit_transform(X)
pca=PCA(n_components=2)
x_pca=pca.fit_transform(x_standardized)
print(x_pca)
pca_df=pd.DataFrame(data=x_pca,columns=['Principal component 1','Principal component 2'])
pca_df['Target']=y
plt.figure(figsize=(8,6))
colors = ['red','green','blue']
for target,color,label in zip(range(len(target_names)),colors,target_names):
  plt.scatter(
      pca_df.loc[pca_df['Target']==target,'Principal component 1'],pca_df.loc[pca_df['Target']==target,'Principal component 2'],alpha=0.6,color=color,label=label
  )
plt.title('PCA of iris dataset(2 components)')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.legend(title='Target',loc='best')
plt.grid(alpha=0.5)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
iris = load_iris()

df = pd.DataFrame(iris["data"], columns= iris["feature_names"])
df["target"] = iris["target"]
df.to_csv("iris.csv", encoding="utf8", index= False)

plt.figure(figsize=(10, 6))
sns.heatmap(df.astype("float").corr(), cmap="YlGn", annot=True)

from sklearn.model_selection import train_test_split
# train_test_split -> 特徵90%, 特徵10%, 目標90%, 目標10%
x_train, x_test, y_train, y_test = train_test_split(df.drop(["target"], axis=1, df['target'], test_size=0.1))


plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

df = pd.read_csv("googleplaystore.csv")
print(df.head(5))
print(df.info())

df["Last Updated"] = pd.to_datetime(df["Last Updated"], errors="coerce")

print(df.isna().sum())
df["Rating"] = df["Rating"].fillna(df["Rating"].median())
print(df.isna().sum())
df = df.dropna()
print(df.isna().sum())
print(df.shape)


print(df.info())

def extractsize(data):
    try:
        data = data[: len(data)-1]
        data = float(data)
        return data
    except:
        return np.NaN

df["Size"] = df["Size"].map(extractsize)


def extractinstalls(data):
    try:
        data = str(data[: len(data)-1])
        data = data.replace(",", "")
        data = float(data)
        return data
    except:
        return 10000.0

df["Installs"] = df["Installs"].map(extractinstalls)


def review(data):
    try:
        data = float(data)
        return data
    except:
        return np.NaN
    
df["Reviews"] = df["Reviews"].map(review)


def androidversion(data):
    try:
        data = str(data[:3])
        data = float(data)
        return data
    except:
        return 4.4

df["Android Ver"] = df["Android Ver"].map(androidversion)


def price(data):
    try:
        data = str(data)
        data = data.replace("$", "")
        data = float(data)
        return data
    except:
        return 0.0
    
df["Price"] = df["Price"].map(price)
print(df.info())

df["Size"] = df["Size"].fillna(df["Size"].median())

print(df["Type"].unique())
df["Type"] = df["Type"].replace({"Free": 0, "Paid": 1})
print(df.info())


encoder = LabelEncoder()
df["Category"] = encoder.fit_transform(df["Category"])
df["Content Rating"] = encoder.fit_transform(df["Content Rating"])
print(df.info())

df["Year"] = df["Last Updated"].dt.year
df["Month"] = df["Last Updated"].dt.month
df["Day"] = df["Last Updated"].dt.day
print(df.head(5))

df = df.drop(columns=["App", "Genres", "Last Updated", "Current Ver"])
print(df.head(5))


cols = list(df.columns)
for x in cols:
    plt.figure(figsize=(7, 7))
    plt.xlabel(x)
    plt.ylabel("Rating")
    plt.scatter(df[x], df["Rating"], color="brown", marker="o")
    plt.show()

sns.pairplot(df)
plt.show()

plt.figure(figsize=(12, 12))
sns.heatmap(df.corr(), annot=True, cmap="bone")
plt.show()

cor = df.corr()
print(cor["Rating"].sort_values(ascending=True))


X = df.drop("Rating", axis=1)
y = df.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = RandomForestRegressor(n_estimators=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test).round(1)

print(model.score(X_test, y_pred.reshape(-1, 1)))
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

with open("googleplayrating.pkl", "wb")as file:
    pickle.dump(model, file)
    file.close()

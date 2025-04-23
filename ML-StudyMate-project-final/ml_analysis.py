import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("performance_log.csv", names=["timestamp", "qnum", "user", "correct", "is_correct"])

# Classification: Predict if answer will be correct
le = LabelEncoder()
df['user_encoded'] = le.fit_transform(df['user'])
X = df[['qnum', 'user_encoded']]
y = df['is_correct']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = LogisticRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(f"✅ Classification Accuracy: {accuracy*100:.2f}%")

# Clustering: Group users based on correctness
kmeans = KMeans(n_clusters=2)
df['cluster'] = kmeans.fit_predict(df[['qnum', 'is_correct']])
df.to_csv("clustered_output.csv", index=False)

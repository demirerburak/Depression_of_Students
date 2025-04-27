import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("student_depression_dataset.csv")
df = data
pd.set_option('display.max_columns', None)
id = df['id']
df = data.drop(columns=['id','City','Profession','Work Pressure','Job Satisfaction','Degree'])
print(df.columns)
le = preprocessing.LabelEncoder()
label = ['Gender','Family History of Mental Illness','Dietary Habits','Have you ever had suicidal thoughts ?','Sleep Duration']
df.replace('?', np.nan, inplace=True)
for column in df.columns:
        mode= df[column].mode()[0]
        df[column] = df[column].fillna(mode)    
for column in label:
        df[column] = le.fit_transform(df[column])
print(df.head())
x = df.drop(columns=['Depression'])
y = df['Depression']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = model.score(X_test_scaled, y_test)
print("Accuracy:", accuracy)
y_results = model.predict(x)
submission = pd.DataFrame({'id': id,'Actual': y,'Predicted': y_results})
submission.to_csv('submission.csv', index=False)
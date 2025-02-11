import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/customer_churn.csv')

df_select = df[["Churn", "Age", "Years", "Num_Sites", "Account_Manager"]]
X = df_select.drop('Churn', axis=1)
y = df_select['Churn']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilités de churn


print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

with open("data/churn_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Capstone-Project-HR-analytics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load Data
file_path = "hr_analytics (1).csv"
df = pd.read_csv(file_path)
df.head()

# Exploratory Data Analysis (EDA)
print(df.info())
print(df.describe(include='all'))

# Attrition distribution
sns.countplot(data=df, x='Attrition')
plt.title("Attrition Count")
plt.show()

# Correlation Heatmap (only numeric columns)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlations")
plt.show()

# 💡 Step 4: Data Preprocessing
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])

# Encode all other categorical columns except target
cat_cols = df.select_dtypes(include='object').columns.drop(['EmployeeID'])
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop(['Attrition', 'EmployeeID'], axis=1)
y = df['Attrition'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

importances = pd.Series(clf.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()

X_test_copy = X_test.copy()
X_test_copy['PredictedAttrition'] = y_pred
X_test_copy['ActualAttrition'] = y_test.values
X_test_copy.to_csv("HR_Attrition_Predictions.csv", index=False)

print("Project Completed. Results saved to HR_Attrition_Predictions.csv")

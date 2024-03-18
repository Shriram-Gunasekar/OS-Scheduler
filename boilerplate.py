import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Assuming you have a DataFrame 'tasks_df' with the necessary features and target (scheduling outcome)
# Perform feature engineering
X = tasks_df[['CPU_burst_time', 'arrival_time', 'priority', 'task_type']]
y = tasks_df['scheduling_outcome']

# Encode categorical variables
label_encoder = LabelEncoder()
X['task_type'] = label_encoder.fit_transform(X['task_type'])

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate a classification report and confusion matrix
print(classification_report(y_test, y_pred))
plot_confusion_matrix(clf, X_test, y_test, cmap='Blues')


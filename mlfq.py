import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Data Generation (Simulated Data)
np.random.seed(42)  # For reproducibility

# Simulate task characteristics
num_tasks = 1000
cpu_burst_time = np.random.randint(1, 21, num_tasks)  # CPU burst time in milliseconds
priority = np.random.randint(1, 6, num_tasks)  # Priority levels
arrival_time = np.random.randint(0, 100, num_tasks)  # Arrival time in milliseconds

# Simulate MLFQ scheduling outcomes (0 for not selected, 1 for selected)
mlfq_selection = np.random.choice([0, 1], num_tasks, p=[0.8, 0.2])  

# Create a DataFrame from the simulated data
tasks_df = pd.DataFrame({
    'CPU_burst_time': cpu_burst_time,
    'Priority': priority,
    'Arrival_time': arrival_time,
    'MLFQ_selection': mlfq_selection
})

# Step 2: Feature Engineering
# Calculate additional features like remaining execution time, waiting time, etc.
tasks_df['Remaining_time'] = tasks_df['CPU_burst_time']
tasks_df['Waiting_time'] = 0

# Step 3: Model Selection (Random Forest Classifier)
# Define features and target variable
X = tasks_df[['CPU_burst_time', 'Priority', 'Arrival_time', 'Remaining_time', 'Waiting_time']]
y = tasks_df['MLFQ_selection']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 4: Model Training and Evaluation
# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate a classification report and confusion matrix
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

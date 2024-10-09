import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dia = pd.read_csv('diabetes.csv')


#Normalizing
normal= dia.to_numpy()
mean = np.mean(normal)
std_dev = np.std(normal)
standardized_data = (normal - mean) / std_dev


X = np.delete(normal, 0, 1)
y=dia['Condition']


X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 20)
#The number of trees in the forest.
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=50)

# Train
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
check = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
print(check)

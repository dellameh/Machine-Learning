import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import LinearSVC
import seaborn as sns
from sklearn.model_selection import train_test_split

dia = pd.read_csv('diabetes.csv')


#Normalizing
normal= dia.to_numpy()
mean = np.mean(normal)
std_dev = np.std(normal)
standardized_data = (normal - mean) / std_dev


X = np.delete(normal, 0, 1)
y=dia['Condition']
# Features for 2 dimensional plot
#X=dia[['Glucose','َInsulin']]
#X=X.to_numpy()



X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 10)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred })
print(diff)

# 2 dimensional plot
# def plot_svc_decision_function(model, ax=None, plot_support=True):
#     #Plot the decision function for a 2D SVC
#     if ax is None:
#         ax = plt.gca()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
    
#     # create grid to evaluate model
#     x = np.linspace(xlim[0], xlim[1], 30)
#     y = np.linspace(ylim[0], ylim[1], 30)
#     Y, X = np.meshgrid(y, x)
#     xy = np.vstack([X.ravel(), Y.ravel()]).T
#     P = model.decision_function(xy).reshape(X.shape)
    
#     # plot decision boundary and margins
#     ax.contour(X, Y, P, colors='k',
#                levels=[-1, 0, 1], alpha=0.5,
#                linestyles=['--', '-', '--'])
    
#     # plot support vectors
#     if plot_support:
#         ax.scatter(model.support_vectors_[:, 0],
#                    model.support_vectors_[:, 1],
#                    s=300, linewidth=1, facecolors='none')
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
#     plt.show()

# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plot_svc_decision_function(clf)
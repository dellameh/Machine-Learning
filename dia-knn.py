import numpy as np
import pandas as pd
import pandas as pd # data processing
#from termcolor import colored as cl # elegant printing of text
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier

dia = pd.read_csv('diabetes.csv')


#Normalizing
normal= dia.to_numpy()
mean = np.mean(normal)
std_dev = np.std(normal)
standardized_data = (normal - mean) / std_dev


X = np.delete(normal, 0, 1)
y=dia['Condition']


X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 20)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)
scr=knn.score(X_test,y_test)
print("Accuray for your model is ",scr)

res=knn.predict(X_test)
res=res.tolist()
li=[]
for i in res:
    if i==1:
        li.append("positive")
    else:
        li.append("negative")

diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': res, 'predicted as':li  })
print(diff)


sb.scatterplot(x=dia['Insulin'],y=dia['Glucose'], hue=dia['Condition'])
plt.show()
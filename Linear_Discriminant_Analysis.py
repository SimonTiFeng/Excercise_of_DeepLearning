import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score


covertype = pd.read_csv('Excercise_of_MachineLearning_Zhouzhihua/covtype.data.gz', compression='gzip')

y = covertype.iloc[:, -1]  
X = covertype.iloc[:, :-1]

X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.3, random_state=42)  
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val) 

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
y_pred = lda.predict(X_val)
accuracy = accuracy_score(Y_val, y_pred)
print(f'准确率: {accuracy}')

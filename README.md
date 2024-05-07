# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to predict the value.
4. Find the accuracy and display the result.


## Program:
```
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: M. R. ANUMITHA 
RegisterNumber:  212223040018
*/
```
``
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
df= pd.read_csv('/content/spam.csv',encoding= 'ISO-8859-1')
df.head()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['v2'])
y = df['v1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))

```
``
## Output:
![Screenshot 2024-05-07 095137](https://github.com/anumitha2005/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155522855/cfddeeea-6146-43db-8ed4-cba635fae72d)

![Screenshot 2024-05-07 095151](https://github.com/anumitha2005/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155522855/62cb617e-4762-4685-9fad-d61424bd957f)

![Screenshot 2024-05-07 095205](https://github.com/anumitha2005/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155522855/0d455286-faec-4100-ad4c-95cf194ffdd8)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

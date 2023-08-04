import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.utils import Bunch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

data = Bunch(data=[], target=[])
df = pd.read_csv("C:/Users/AMAN PANUILY/OneDrive/Desktop/python/data.csv")

data.data = df.iloc[:, :31].values.tolist()
data.target = df[df.columns[len(df.columns) - 1]].tolist()

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=20)

model = MLPClassifier()
model.fit(x_train, y_train)

# f1 score , recall score, precision , confusion matrix, accuracy
# make prediction
y_pred = model.predict(x_test)
f = open("mlp.p", 'wb')
pickle.dump(model, f)
f.close()




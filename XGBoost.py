import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import  Bunch
import pickle
data = Bunch(data=[], target= [])
df = pd.read_csv("C:/Users/AMAN PANUILY/OneDrive/Desktop/python/data.csv")

data.data = df.iloc[:,:31].values.tolist()
data.target = df[df.columns[len(df.columns)-1]].tolist()

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target,test_size=0.3,random_state=20)

model = xgb.XGBClassifier()
model.fit(x_train,y_train)

# make prediction
y_pred = model.predict(x_test)
f = open("XGBoost.p", 'wb')
pickle.dump(model, f)
f.close()
print('done')
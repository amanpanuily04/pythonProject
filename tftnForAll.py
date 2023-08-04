import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import  Bunch
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
#xgboost
f=open("XGboost.p",'rb')
model = pickle.load(f)
data = Bunch(data=[], target= [])
df = pd.read_csv("C:/Users/AMAN PANUILY/OneDrive/Desktop/python/data.csv")
data.data = df.iloc[:,:31].values.tolist()
data.target = df[df.columns[len(df.columns)-1]].tolist()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target,test_size=0.2,random_state=20)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)*100
tn, fp, fn, tp=confusion_matrix(y_test,y_pred).ravel()
# print("accuracy for XGBoost is= %.2f" % accuracy)
print("true negative = %d "%tn)
print("false positive = %d "%fp)
print("false negative = %d"%fn)
print("true positive = %d"%tp)
new_accuracy = (tp+tn)/(tp+tn+fp+fn)
new_accuracy = new_accuracy*100
print("new accuracy for XGBoost is= %.2f" % new_accuracy)
precision = tp/(tp+fp)
precision = precision*100
print("precision = %.2f" % precision)
recall= tp/(tp+fn)
recall = recall*100
print("recall = %.2f" % recall)
F1_score = (2*precision*recall)/(precision + recall);
print("f1_score= %.2f" % F1_score);
f.close()

#rf
f=open("random_forest.p",'rb')
model = pickle.load(f)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)*100
tn, fp, fn, tp=confusion_matrix(y_test,y_pred).ravel()
# print("accuracy for random forest is= %.2f" % accuracy)
print("true negative = %d "%tn)
print("false positive = %d "%fp)
print("false negative = %d"%fn)
print("true positive = %d"%tp)
new_accuracy = (tp+tn)/(tp+tn+fp+fn)
new_accuracy = new_accuracy*100
print("new accuracy for Random forest is= %.2f" % new_accuracy)
precision = tp/(tp+fp)
precision = precision*100
print("precision = %.2f" % precision)
recall= tp/(tp+fn)
recall = recall*100
print("recall = %.2f" % recall)
F1_score = (2*precision*recall)/(precision + recall);
print("f1_score= %.2f" % F1_score);
f.close()

#ridge
f=open("ridge.p",'rb')
model = pickle.load(f)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)*100
tn, fp, fn, tp=confusion_matrix(y_test,y_pred).ravel()
# print("accuracy for Ridge is= %.2f" % accuracy)
print("true negative = %d "%tn)
print("false positive = %d "%fp)
print("false negative = %d"%fn)
print("true positive = %d"%tp)
new_accuracy = (tp+tn)/(tp+tn+fp+fn)
new_accuracy = new_accuracy*100
print("new accuracy for Ridge is= %.2f" % new_accuracy)
precision = tp/(tp+fp)
precision = precision*100
print("precision = %.2f" % precision)
recall= tp/(tp+fn)
recall = recall*100
print("recall = %.2f" % recall)
F1_score = (2*precision*recall)/(precision + recall);
print("f1_score= %.2f" % F1_score);
f.close()

#decision tree
f= open("decisiontree.p",'rb')
model = pickle.load(f)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred) * 100
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# print("accuracy for Decision tree is= %.2f" % accuracy)
print("true negative = %d " % tn)
print("false positive = %d " % fp)
print("false negative = %d" % fn)
print("true positive = %d" % tp)


new_accuracy = (tp+tn)/(tp+tn+fp+fn)
new_accuracy = new_accuracy*100
print("new accuracy for Decision tree is= %.2f" % new_accuracy)

precision = tp/(tp+fp)
precision = precision*100
print("precision = %.2f" % precision)

recall= tp/(tp+fn)
recall = recall*100
print("recall = %.2f" % recall)

F1_score = (2*precision*recall)/(precision + recall);
print("f1_score= %.2f" % F1_score);
f.close()

#mlp

f = open("mlp.p",'rb')
model = pickle.load(f)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred) * 100
tp, tn, fp, fn = confusion_matrix(y_test, y_pred).ravel()
# print("accuracy for MLP is= %.2f" % accuracy)
print("true negative = %d " % tn)
print("false positive = %d " % fp)
print("false negative = %d" % fn)
print("true positive = %d" % tp)

new_accuracy = (tp + tn) / (tp + tn + fp + fn)
new_accuracy = new_accuracy * 100
print("new accuracy for MLP is= %.2f" % new_accuracy)

precision = tp / (tp + fp)
precision = precision * 100
print("precision = %.2f" % precision)

recall = tp / (tp + fn)
recall = recall * 100
print("recall = %.2f" % recall)

F1_score = (2 * precision * recall) / (precision + recall)
print("f1_score= %.2f" % F1_score)
f.close()
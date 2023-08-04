# roc curve and auc
import pickle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

data = Bunch(data=[], target= [])
df = pd.read_csv("C:/Users/AMAN PANUILY/OneDrive/Desktop/python/data.csv")


data.data = df.iloc[:,:31].values.tolist()
data.target = df[df.columns[len(df.columns)-1]].tolist()

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target,test_size=0.3,random_state=20)

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# load a model rf
mdl = open("random_forest.p", "rb")
model = pickle.load(mdl)
lr_probs = model.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
# lr_auc = roc_auc_score(y_test, lr_probs)

threshold = 0.8
y_pred_threshold = np.where(lr_probs >= threshold, 1, 0)
auc_threshold = roc_auc_score(y_test, y_pred_threshold)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Randoom forest  ROC AUC=%.3f' % (auc_threshold))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred_threshold)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill: ROC AUC=%.3f' % (ns_auc))
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Random forest : ROC AUC=%.3f' % (auc_threshold))
# axis labels

# load a model xgboost
mdl = open("XGBoost.p", "rb")
model = pickle.load(mdl)
lr_probs = model.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
# lr_auc = roc_auc_score(y_test, lr_probs)
threshold = 0.5
y_pred_threshold = np.where(lr_probs >= threshold, 1, 0)
auc_threshold = roc_auc_score(y_test, y_pred_threshold)


# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('xgBOOST AUC=%.3f' % (auc_threshold))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred_threshold)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill: ROC AUC=%.3f' % (ns_auc))
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='xgBOOST : ROC AUC=%.3f' % (auc_threshold))

# ridge

# load a model ridge
mdl = open("ridge.p","rb")
model = pickle.load(mdl)
lr_probs = model._predict_proba_lr(x_test)

# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
# lr_auc = roc_auc_score(y_test, lr_probs)

threshold = 0.4
y_pred_threshold = np.where(lr_probs >= threshold, 1, 0)
auc_threshold = roc_auc_score(y_test, y_pred_threshold)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('ridge  ROC AUC=%.3f' % (auc_threshold))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred_threshold)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill: ROC AUC=%.3f' % (ns_auc))
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='ridge : ROC AUC=%.3f' % (auc_threshold))

# decision


# load a model decision tree
mdl = open("decisiontree.p","rb")
model = pickle.load(mdl)
lr_probs = model.predict_proba(x_test)

# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('decisiontree  ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill: ROC AUC=%.3f' % (ns_auc))
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='decisiontree : ROC AUC=%.3f' % (lr_auc))


#mlp

# load a model mlp
mdl = open("mlp.p", "rb")
model = pickle.load(mdl)
lr_probs = model.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('mlp AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill: ROC AUC=%.3f' % (ns_auc))
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='mlp : ROC AUC=%.3f' % (lr_auc))



# axis labels
pyplot.xlabel('False Positive Rate ')
pyplot.ylabel('True Positive Rate')
# show the legend and title
pyplot.legend()
pyplot.title("ROC curve")

# show the plot
pyplot.show()
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas.core.common import random_state
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm.notebook import tqdm

sns.set_style('whitegrid')

from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
   print('User uploaded file «{name}» with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))
   
dataframe = pd.read_excel("ONLINEADS.xlsx")
dataframe.info()
dataframe = dataframe.drop(columns =["Client ID"])
repeated_values = dataframe[dataframe.duplicated()]
repeated_values.count()
dataframe1 = dataframe.drop_duplicates(keep = 'first')
repeated_values = dataframe1[dataframe1.duplicated()]
repeated_values.count()

for column in dataframe.columns:
  sns.displot(data=dataframe[column], discrete=True)
  
ax = sns.heatmap(dataframe.corr())

sns.violinplot(x="Registration", y = "Age", data=dataframe)
sns.violinplot(x="Registration", y = "Interest", data=dataframe)
sns.violinplot(x="Registration", y = "Device", data=dataframe)
sns.violinplot(x="Registration", y = "AdsTool", data=dataframe)

y = dataframe1["Registration"]
X = dataframe1.drop('Registration', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

dataframe1['VisitTime'] = dataframe1['VisitTime'].div(10)
clf = KNeighborsClassifier(n_neighbors=10)
clf = clf.fit(X_train, y_train)
train_predict = clf.predict(X_test)
clf.score(X_test, y_test)
training_accuracy = []
test_accuracy = []
# пробуем n_neighbors от 1 до 10
neighbors_settings = range(1, 20)
for n_neighbors in neighbors_settings:
# строим модель
  clf = KNeighborsClassifier(n_neighbors=n_neighbors)
  clf.fit(X_train, y_train)
  # записываем правильность на обучающем наборе
  training_accuracy.append(clf.score(X_train, y_train))
  # записываем правильность на тестовом наборе
  
  
  test_accuracy.append(clf.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="правильность на обучающем наборе")
plt.plot(neighbors_settings, test_accuracy, label="правильность на тестовом наборе")
plt.ylabel("Правильность")
plt.xlabel("количество соседей")
plt.legend()

log_reg = LogisticRegression(max_iter=1000)
log_reg = log_reg.fit(X_train, y_train)
train_predict = log_reg.predict_proba(X_train)
test_predict = log_reg.predict_proba(X_test)

test_predict[:10, :]
train_predict = train_predict[:, 1]
test_predict = test_predict[:, 1]
plt.figure(figsize=(20, 5))

plt.hist(train_predict, bins=100)
plt.vlines(0.3, 0, 2000)

score = log_reg.score(X_train, y_train)
score

score = log_reg.score(X_test, y_test)
score

test_predict = log_reg.predict(X_test)
log_reg_conf_matrix = confusion_matrix(y_test, test_predict)
log_reg_conf_matrix = pd.DataFrame(log_reg_conf_matrix)

log_reg_conf_matrix

false_p, true_p, threshold = roc_curve(y_test, test_predict)
plt.figure(figsize=(20, 7))
plt.plot(false_p, true_p, label="Сглаженные значения ROC-AUC")
plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle=':')
plt.fill_between(false_p, true_p, step='mid', alpha=0.4)
plt.legend()
plt.show()

roc_auc_value = roc_auc_score(y_test, test_predict)
roc_auc_value

clf = DecisionTreeClassifier() # создаем классификатор

clf = clf.fit(X_train, y_train) # обучаем

y_pred = clf.predict(X_test) # предсказание

print("Accuracy on training set: {:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))

clf = DecisionTreeClassifier(criterion="entropy", max_depth=6)

clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy on training set: {:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))

#Визуализация
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled = True, rounded = True,
                special_characters = True, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('ads.png')
Image(graph.create_png())

clf = RandomForestClassifier(n_estimators=100, max_features="sqrt", max_depth=6, min_samples_split=2, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy on training set: {:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))

X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled, X_test_scaled = [
    scaler.transform(feature_matrix)
    for feature_matrix in (X_train_scaled, X_test_scaled)
]
clf = KNeighborsClassifier(n_neighbors=10)
clf = clf.fit(X_train_scaled, y_train)

clf.score(X_test_scaled, y_test)
y_train_predicted = log_reg.predict(X_train_scaled)
y_test_predicted = log_reg.predict(X_test_scaled)

from sklearn.metrics import f1_score, accuracy_score


def calculate_metrics(y_train, y_train_predicted, y_test, y_test_predicted):
    metics_pd = pd.DataFrame({
        "accuracy": [accuracy_score(y_train, y_train_predicted), 
                     accuracy_score(y_test, y_test_predicted)],
    }, index=["train", "test"])

    # for average in ['macro', 'micro', 'weighted']:
    for average in ['macro']:
        metics_pd[f"f1-{average}"] = [
            f1_score(y_train, y_train_predicted, average=average),
            f1_score(y_test, y_test_predicted, average=average)
        ]

    return metics_pd
    
calculate_metrics(y_train, y_train_predicted, y_test, y_test_predicted)
log_reg = LogisticRegression(max_iter=1000)
log_reg = log_reg.fit(X_train_scaled, y_train)

score = log_reg.score(X_train_scaled, y_train)
score

score = log_reg.score(X_test_scaled, y_test)
score

y_train_predicted = log_reg.predict(X_train_scaled)
y_test_predicted = log_reg.predict(X_test_scaled)

calculate_metrics(y_train, y_train_predicted, y_test, y_test_predicted)

y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="kNN, auc="+str(auc))

y_pred_proba = log_reg.predict_proba(X_test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="LogReg, auc="+str(auc))

plt.legend(loc=4)
plt.show()

models = dict(
    KNeighborsClassifier=KNeighborsClassifier(n_neighbors=10),
    LogisticRegression=LogisticRegression(max_iter=1000),
)

N_PARTS = 10
part_size = round(X_train_scaled.shape[0] / N_PARTS)
part_sizes = list(range(
    part_size, X_train_scaled.shape[0] + part_size, part_size
))
train_test_subsets = tuple(zip(
    [X_train_scaled[0:size] for size in part_sizes],
    [y_train[0:size] for size in part_sizes],
    [X_test_scaled] * N_PARTS,
    [y_test] * N_PARTS
))

np.diff(part_sizes)
global_scores = defaultdict(lambda: defaultdict(list))

for model_name, model in tqdm(models.items(), desc='outer'):
    for X_train_subset, y_train_subset, X_test, y_test in train_test_subsets:
        model.fit(X_train_subset, y_train_subset)
        y_train_predicted = model.predict(X_train_subset)
        y_test_predicted = model.predict(X_test)
        
        global_scores[model_name]["f1-macro"].append(f1_score(y_test, y_test_predicted, average="macro"))
        global_scores[model_name]["accuracy"].append(accuracy_score(y_test, y_test_predicted))
        
global_scores
def plot_learning_curve(fig, ax, train_subsets_sizes, scores, 
                        metric_names, colors, suptitle, caption,
                        xlabel, ylabel="metric"):
    fig.subplots_adjust(bottom=0.2, top=0.85, wspace=0.1)
    fig.suptitle(suptitle, fontsize=25)
    fig.align_labels(axs=ax)
    fig.text(0.5, 0.025, caption, wrap=True, horizontalalignment='center', fontsize=24)

    for i, metric_name in enumerate(metric_names):
        ax[i].set_title(f"{metric_name} in Test sample", fontsize=22, pad=10)
        ax[i].set_xlabel(xlabel, fontsize=22, labelpad=10)
        ax[i].set_ylabel(ylabel, fontsize=22, labelpad=10)
        ax[i].set_xticks(train_subsets_sizes)
        ax[i].ticklabel_format(style='sci', axis='y')
    ax[0].yaxis.set_label_position("left")
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()

    for model_name, model_scores in scores.items():
        for i, (metric, scores) in enumerate(model_scores.items()):
            print(model_name, i, metric, scores)
            ax[i].plot(
                train_subsets_sizes, scores, 
                label=model_name, 
                color=colors[model_name],
                marker="*", 
                linewidth=2, 
                markersize=16,
            )
            
            legend = ax[i].legend(loc="bottom right", fontsize=15)
            frame = legend.get_frame()
            frame.set_facecolor("white")
            frame.set_edgecolor("black")

    plt.show()
len(part_sizes), len(global_scores['KNeighborsClassifier']['accuracy'])
fig, ax = plt.subplots(1, 2, figsize=(25, 10))
model_names = list(models.keys())
metric_names = list(list(global_scores.values())[0].keys())

plot_learning_curve(fig, ax, part_sizes[:-1], global_scores,
                    metric_names=metric_names,
                    colors={model_names[0]: "xkcd:azure",
                            model_names[1]: "xkcd:tangerine"},
                    suptitle="Learning curve",
                    caption="How does the model behaviour change with "
                            "the increase of the train dataset size.",
                    xlabel="train dataset size")

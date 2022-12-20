import os
import json
import gzip
import numpy as np
import matplotlib.pyplot as plt
from properties import datafolder, resultsdatafolder
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

with open(os.path.join(datafolder, "6_optimal_num_topics.json")) as infile:
	optimal_num_topics = json.load(infile)

num_assignees = 5
projects = ["AMBARI", "ARROW", "CASSANDRA", "CB", "DATALAB", "FLINK", "GEODE", "HDDS", "IGNITE", "IMPALA", "MESOS", "OAK"]
model_keys = ['labels', 'labels_top_terms']
model_titles = ['Simple Labels', 'Enhanced Labels']

# Produce Figures 7 and 8 of paper
accuracy_simple, fmeasure_simple = [], []
accuracy_enhanced, fmeasure_enhanced = [], []
for project in projects:
	with gzip.open(os.path.join(datafolder, "5_" + project + "_" + str(num_assignees) + "_assignees" + "_results.json.gz"), 'r') as infile:
		results = json.loads(infile.read().decode('utf-8'))
	yclasses = results["classes"]
	ytest = results["y_test"]
	for model_key in model_keys:
		num_topics = optimal_num_topics[project + "_" + str(num_assignees) + "_assignees"]
		y_pred = results[str(num_topics)]["SVM"][model_key]["y_pred"]
		y_score = results[str(num_topics)]["SVM"][model_key]["y_pred_proba"]
		label_binarizer = LabelBinarizer().fit(yclasses)
		y_true = label_binarizer.transform(ytest)
		y_pred = label_binarizer.transform(y_pred)
		if model_key == 'labels':
			accuracy_simple.append(accuracy_score(y_true, y_pred))
			fmeasure_simple.append(f1_score(y_true, y_pred, average='micro'))
		else:
			accuracy_enhanced.append(accuracy_score(y_true, y_pred))
			fmeasure_enhanced.append(f1_score(y_true, y_pred, average='micro'))

index = np.arange(len(projects))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(4.4, 3.0))
rects1 = plt.bar(index, accuracy_simple, bar_width, label='Simple Labels')
rects2 = plt.bar(index + bar_width, accuracy_enhanced, bar_width, label='Enhanced Labels')
ax.set_ylim(0, 0.94)
#ax.set_xlabel('Projects')
ax.set_ylabel('Accuracy')
ax.set_xticks(index + bar_width/2, projects, rotation = 45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(resultsdatafolder, "AccuracyLabels.eps"))
plt.savefig(os.path.join(resultsdatafolder, "AccuracyLabels.pdf"))

fmeasure_simple = [0.22, 0.09, 0.6, 0.49, 0.64, 0.27, 0.24, 0.23, 0.68, 0.47, 0.49, 0.4]
fmeasure_enhanced = [0.46, 0.32, 0.62, 0.49, 0.65, 0.39, 0.28, 0.33, 0.72, 0.51, 0.53, 0.47]
fig, ax = plt.subplots(figsize=(4.4, 3.0))
rects1 = plt.bar(index, fmeasure_simple, bar_width, label='Simple Labels')
rects2 = plt.bar(index + bar_width, fmeasure_enhanced, bar_width, label='Enhanced Labels')
ax.set_ylim(0, 0.94)
#ax.set_xlabel('Projects')
ax.set_ylabel('F-measure')
ax.set_xticks(index + bar_width/2, projects, rotation = 45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(resultsdatafolder, "FmeasureLabels.eps"))
plt.savefig(os.path.join(resultsdatafolder, "FmeasureLabels.pdf"))
#plt.show()


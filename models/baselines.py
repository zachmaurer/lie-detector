from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import numpy as np
import json

"""
Assumes features and labels are json files as created by feature_extractor.py and in the feature_extraction directory.

For baseline result, trains svm on feature averages over time (I know this is really dumb but hey it's a baseline right?) and does cross-validation.

"""
avg_features = []
labels = []
with open('../feature_extraction/extracted_features.json', 'r') as feature_file:
	features = json.load(feature_file)
	for segment in features:
		segment = np.array(segment)
		avg_features.append(np.mean(segment,axis=1))

avg_features = np.stack(avg_features)
with open('../feature_extraction/labels.json', 'r') as label_file:
	labels = np.array(json.load(label_file))


linear_svm = svm.SVC(kernel='linear', C=1)
#Cross validation scores were .779 for this speaker
#scores = cross_val_score(linear_svm, avg_features, labels, cv=5)
#print scores       
linear_svm.fit(avg_features[:50,:],labels[:50])
preds = linear_svm.predict(avg_features[50:,:])
print classification_report(labels[50:],preds)
#precision for lies and recall for truths were both 1.0, but recall for lies was .2 and precision for truths was .76. f1 score for lies was .33. Obvious room for improvement.


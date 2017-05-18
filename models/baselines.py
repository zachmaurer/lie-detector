from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from scipy.stats import skew
import numpy as np
import json

"""
Assumes features and labels are json files as created by feature_extractor.py and in the feature_extraction directory.

For baseline result, trains svm on feature averages, maxs, mins, std devs and skews across time.

"""
summarized_features = []
labels = []
with open('../feature_extraction/extracted_features.json', 'r') as feature_file:
	time_series_features = json.load(feature_file)
	for segment in time_series_features:
		segment = np.array(segment)
		avgs = np.mean(segment,axis=1)
		maxs = np.max(segment,axis=1)
		mins = np.min(segment,axis=1)
		skews = skew(segment,axis=1)
		std = np.std(segment,axis=1)
		summary = np.hstack([avgs,maxs,mins,skews,std])
		summarized_features.append(summary)


summarized_features = np.stack(summarized_features)
print summarized_features.shape


with open('../feature_extraction/labels.json', 'r') as label_file:
	labels = np.array(json.load(label_file))

#split data into training and testing
indices = np.random.permutation(summarized_features.shape[0])
#this decides the percentage of the data in the training set
lastTraining = 9*len(summarized_features)/10
training_idx, test_idx = indices[:lastTraining], indices[lastTraining:]
training_x, test_x = summarized_features[training_idx,:], summarized_features[test_idx,:]
training_y, test_y = labels[training_idx], labels[test_idx]


gammas = np.logspace(-2, -1, 20)
for gamma in gammas:
	linear_svm = svm.SVC(C=1, gamma=gamma)
	linear_svm.fit(training_x,training_y)
	preds = linear_svm.predict(test_x)
	print "gamma= " + str(gamma)
	print "accuracy = " + str(np.sum(preds == test_y)/float(len(preds)))
	print classification_report(test_y,preds)
#precision for lies and recall for truths were both 1.0, but recall for lies was .2 and precision for truths was .76. f1 score for lies was .33. Obvious room for improvement.


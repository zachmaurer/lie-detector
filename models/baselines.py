from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import skew
import numpy as np
import json

"""
Assumes features and labels are json files as created by feature_extractor.py and in the feature_extraction directory.

For baseline result, trains svm on feature averages, maxs, mins, std devs and skews across time.

"""
summarized_features = []
labels = []
summarized_features_by_id = {}
labels_by_id = {}
with open('../feature_extraction/extracted_features_0.05_0.025.json', 'r') as feature_file:
	with open('../feature_extraction/labels_0.05_0.025.json', 'r') as label_file:
		time_series_features = json.load(feature_file)
		loaded_labels = json.load(label_file)
		for filename, segment in time_series_features.iteritems():
			label = loaded_labels[filename]
			subject_id = filename[2:4]
			if subject_id not in labels_by_id:
				labels_by_id[subject_id] = []
			labels_by_id[subject_id].append(label)
			labels.append(label)
			segment = np.array(segment)
			avgs = np.mean(segment,axis=1)
			maxs = np.max(segment,axis=1)
			mins = np.min(segment,axis=1)
			skews = skew(segment,axis=1)
			std = np.std(segment,axis=1)
			summary = np.hstack([avgs,maxs,mins,skews,std])
			if subject_id not in summarized_features_by_id:
				summarized_features_by_id[subject_id] = []
			summarized_features_by_id[subject_id].append(summary)
			summarized_features.append(summary)

labels = np.array(labels)

all_preds = []
all_test_y = []
for subject_id, x in summarized_features_by_id.iteritems():
	x = np.stack(summarized_features_by_id[subject_id])
	y = np.array(labels_by_id[subject_id])
	indices = np.random.permutation(len(y))
	lastTraining = 8*len(x)/10
	training_idx, test_idx = indices[:lastTraining], indices[lastTraining:]
	training_x, test_x = x[training_idx,:], x[test_idx,:]
	training_y, test_y = y[training_idx], y[test_idx]
	neigh = KNeighborsClassifier(n_neighbors=6) #default k=5, seems fine to me
	neigh.fit(training_x, training_y) 
	preds = neigh.predict(test_x)
	all_preds.append(preds)
	all_test_y.append(test_y)
	print "results for subject id " + str(subject_id)
	print "accuracy = " + str(np.sum(preds == test_y)/float(len(preds)))
	print classification_report(test_y,preds)
all_preds = np.hstack(all_preds)
all_test_y = np.hstack(all_test_y)
print "results for all subjects:"
print "accuracy = " + str(np.sum(all_preds == all_test_y)/float(len(all_preds)))
print classification_report(all_test_y,all_preds)



"""
#split data into training and testing
indices = np.random.permutation(summarized_features.shape[0])
#this decides the percentage of the data in the training set
lastTraining = 8*len(summarized_features)/10
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
"""
#precision for lies and recall for truths were both 1.0, but recall for lies was .2 and precision for truths was .76. f1 score for lies was .33. Obvious room for improvement.


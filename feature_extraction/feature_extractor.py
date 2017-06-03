from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import os
import json
import ujson
import numpy as np
"""
Run speaker dependent feature extractor first!! 
Extract the features from all the files and save them to a numpy matrix in textform so that we don't have to deal with audio files anymore. Not 
sure if the directory here below is correct, was a little confused about what "aligned" and "pedal" meant.
Assumes converter.py just got run.
Also assumes you have pyAudioAnalysis, you can either add it to your path or just clone it into the feature_extraction directory. 
Follow the installation instructions on the pyAudioAnalysis github https://github.com/tyiannak/pyAudioAnalysis/wiki/2.-General

Output of stFeatureExtraction is described here https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction
"""
labels = {}
features = {}
#Should work after running converter.py
datadir = "../data/processed/"
frame_size = 0.050*2
frame_stepsize = 0.025*2
speaker_file = 'speaker_dependent_features_{}_{}.json'.format(frame_size, frame_stepsize)
with open(speaker_file, 'r') as data:
	speaker_features = ujson.load(data)
print("Frame_size: {},  Step_size: {}".format(frame_size, frame_stepsize))
total = 0
with open(speaker_file, 'r') as data:
	speaker_features = ujson.load(data)
for i, dirname in enumerate(os.listdir(datadir)):
	if dirname == ".DS_Store":
		continue
	speaker = dirname 
	for filename in os.listdir(datadir + dirname + '/audio_trimmed/pedal/'):
		if filename == ".DS_Store":
			continue
		if "lie" in filename:
			#labels.append((filename, 1))
			labels[filename] = 1
		else:
			labels[filename] = 0
		[Fs, x] = audioBasicIO.readAudioFile(datadir+dirname+'/audio_trimmed/pedal/'+filename)
		#we might want to play with the timeframe here - as it is this is giving us up to ~1.5k frames for our sequences
		speaker_feat = speaker_features[dirname]
		st_features = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_size*Fs, frame_stepsize*Fs)
		num_features, num_windows = st_features.shape
		new_features = np.zeros((num_features, num_windows))
		for i in range(num_features):
			new_features[i] = (st_features[i] - speaker_feat[i])/speaker_feat[i]
		st_features = np.concatenate((st_features, new_features))
		features[filename] = st_features.tolist()
		total += 1
	print(i)
print(total)
with open('labels_{}_{}.json'.format(frame_size, frame_stepsize), 'w') as label_file:
	json.dump(labels,label_file)
with open('extracted_features_{}_{}.json'.format(frame_size, frame_stepsize),'w') as feature_file:
	json.dump(features,feature_file)

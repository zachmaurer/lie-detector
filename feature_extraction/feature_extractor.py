from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import os
import json
"""
Extract the features from all the files and save them to a numpy matrix in textform so that we don't have to deal with audio files anymore. Not 
sure if the directory here below is correct, was a little confused about what "aligned" and "pedal" meant.
Assumes structure like Zach had in the version of S-1A he sent us with this sitting in the same directory as S-1A, but easy to switch this up

"""
labels = []
features = []
#I had Zach's example data sitting in the project directory
datadir = "../S-1A/audio/aligned/"
for filename in os.listdir(datadir):
	if "lie" in filename:
		labels.append(1)
	else:
		labels.append(0)
	[Fs, x] = audioBasicIO.readAudioFile(datadir+filename)
	#we might want to play with the timeframe here - as it is this is giving us up to ~1.5k frames for our sequences
	st_features = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
	features.append(st_features.tolist())

with open('labels.json', 'w') as label_file:
	json.dump(labels,label_file)
with open('extracted_features.json','w') as feature_file:
	json.dump(features,feature_file)

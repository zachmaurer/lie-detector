from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import os
import json
"""
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
print("Frame_size: {},  Step_size: {}".format(frame_size, frame_stepsize))
total = 0
for i, dirname in enumerate(os.listdir(datadir)):
	if dirname == ".DS_Store":
		continue
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
		st_features = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_size*Fs, frame_stepsize*Fs)
		features[filename] = st_features.tolist()
		total += 1
	print(i)
print(total)
with open('labels_{}_{}.json'.format(frame_size, frame_stepsize), 'w') as label_file:
	json.dump(labels,label_file)
with open('extracted_features_{}_{}.json'.format(frame_size, frame_stepsize),'w') as feature_file:
	json.dump(features,feature_file)

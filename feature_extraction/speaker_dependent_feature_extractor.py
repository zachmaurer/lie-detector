from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import os
import json
import numpy as np
"""
Run this file first, then feature extractor. 
"""
features = {}
#Should work after running converter.py
datadir = "../data/processed/"
frame_size = 0.1
frame_stepsize = 0.05
print("Frame_size: {},  Step_size: {}".format(frame_size, frame_stepsize))
total = 0
for i, dirname in enumerate(os.listdir(datadir)):
	if dirname == ".DS_Store":
		continue
	speaker = dirname
	speaker_count = 0
	for filename in os.listdir(datadir + dirname + '/audio_trimmed/pedal/'):
		if filename == ".DS_Store":
			continue
		if "lie" in filename:
			continue
		else:
			
			[Fs, x] = audioBasicIO.readAudioFile(datadir+dirname+'/audio_trimmed/pedal/'+filename)
			#we might want to play with the timeframe here - as it is this is giving us up to ~1.5k frames for our sequences
			audio_features = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_size*Fs, frame_stepsize*Fs)
			audio_features = np.mean(audio_features, axis=1)
			if speaker_count == 0: 
				st_features = audio_features
			else: 
				st_features += audio_features

			speaker_count += 1
			total += 1

	features[speaker] = (st_features/speaker_count).tolist()
	print(i)
print(total)

with open('speaker_dependent_features_{}_{}.json'.format(frame_size, frame_stepsize),'w') as feature_file:
	json.dump(features,feature_file)

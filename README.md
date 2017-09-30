# Neural Lie Detection

Objective:

* Using the CMU Deceptive Speech Corpus (CDC),  develop a neural network that correctly distinguishes lies from truths in interview recordings.

## Summary of Findings

* Cleaning the data was challenging, as it appeared that there were some areas when the audio transcriptions did not exactly align with the recorded audio.
* A greedy algorithm was used to align these fragments and k-means was used determine a loudness theshold to help strip leading and lagging silence from the audio clips.
* Models mainly consisted of a series of LSTMs; the output of which was combined in different ways.
* We also prototyped a model that used a set of stacked, dilated 1D convolutions over the encoded input, roughly inspired by WaveNet.
* Simpler models performed just as well as more complex models.
* The most important factors for increasing performance was the addition of transcript data encoded in GloVe vectors. 
* Previous work on this subject could benefit from (1) better feature selection and (2) more rigorous cross validation techniques for establishing accuracy estimates.
* Previous work only used SVM classifiers on aggregate acoustic measurement features. Essentially, previous work recorded an accuracy that was roughly consistent with the majority class distribution.
* Our work yielded evidence to support that (1)  LSTMs can be used effectively on this task, (2) lexical information appears to be more predictive than acoustic features and (3) using a more rigourous rotating, single speaker test set, our test set accuracy was closer to 78.5% instead of the roughly ~63% accuracy that was previously observed.
* To strengthen these conclusions, it would be necessary to investigate the per-speaker distributions of lies vs. truths.

## Poster

![Poster](https://github.com/zachmaurer/lie-detector/raw/master/assets/cs224s-poster.jpg)

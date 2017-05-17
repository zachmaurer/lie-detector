import argparse
import os
import json
from pydub import AudioSegment
from sklearn.cluster import KMeans
import numpy as np

RAW_DATA_PATH = os.path.abspath("../data/raw")
PROCESSED_DATA_PATH = os.path.abspath("../data/processed")

PEDAL_CORR_HEADER = 18
PEDAL_CORR_CSV = "{}_labels.csv"

PUNC_TEXTGRID_HEADER = 4
PUNC_ALIGNED_CSV = "{}_labels_aligned.csv"
PUNC_ALIGNED_JSON = "{}_labels_aligned.json"

AUDIO_FILE = "{}_R_16k.flac"
PROCESSED_AUDIO = "{}_{}_{}.wav"


# This converter script needs the following file structure to run.
# /project 
#     /cleaning
#     /data
#           /raw
#               /S-1A
#               /S-1B
#               /...
#          /processed

def findIndex(sound, threshold, onset = 100, chunk_size = 10):
  # Source: http://stackoverflow.com/questions/29547218/remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
  trim_ms = 0
  while sound[trim_ms:trim_ms+chunk_size].dBFS < threshold:
      trim_ms += chunk_size
  trim_ms = (trim_ms-onset) if (trim_ms - onset) > 0 else 0
  return trim_ms

def trimSilence(subject, aligned = False):
  mode = "aligned" if aligned else "pedal"
  input_dir = os.path.join(PROCESSED_DATA_PATH, subject, "audio", mode)

  save_dir = os.path.join(PROCESSED_DATA_PATH, subject, "audio_trimmed", mode)
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

  audio_files = [f for f in os.listdir(input_dir) if f[0] != "."]
  sample_size = 10
  for file in audio_files:
    wav = AudioSegment.from_file(os.path.join(input_dir, file), fomat = "wav")
    samples = np.array([ wav[i : i + sample_size].dBFS for i in range(0, len(wav), sample_size)])
    samples = samples.reshape(-1, 1)
    clusters = KMeans(n_clusters=3, random_state=0).fit(samples).cluster_centers_
    clusters = clusters.reshape(3,)
    clusters = sorted(clusters, reverse = True)
    threshold = (clusters[0] - clusters[1])*0.9 + clusters[1]
    start = findIndex(wav, threshold)
    end = findIndex(wav.reverse(), threshold)
    wav[start: len(wav) - end].export(os.path.join(save_dir, file), format = "wav")


def removeArtifacts():
  subjects = [s for s in os.listdir(RAW_DATA_PATH) if s[0] != "."]
  for i, s in enumerate(subjects):
    print("Trimming {}/{}".format(i+1, len(subjects)))
    trimSilence(s)


# function: trimExamples
# ---
# Trims examples on the basis of one of the CSV files
#
def spliceAudio(subject, aligned = True):
  mode = "aligned" if aligned else "pedal"
  save_dir = os.path.join(PROCESSED_DATA_PATH, subject, "audio", mode)
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

  audio_file = os.path.join(RAW_DATA_PATH, subject, AUDIO_FILE.format(subject))
  flac = AudioSegment.from_file(audio_file, fomat = "flac")

  if aligned:
    csv = os.path.join(PROCESSED_DATA_PATH, subject, PUNC_ALIGNED_CSV.format(subject))
  else:
    csv = os.path.join(PROCESSED_DATA_PATH, subject, PEDAL_CORR_CSV.format(subject))

  with open(csv, 'r') as csv:
    for line in csv:
      subject, start, end, label, interval = line.strip(" \n").split(",")
      splice_name = PROCESSED_AUDIO.format(subject, interval, label)
      splice_file = os.path.join(save_dir, splice_name)
      start = float(start) * 1000
      end = float(end) * 1000
      flac[start:end+1].export(splice_file, format = "wav")

def trimExamples():
  subjects = [s for s in os.listdir(RAW_DATA_PATH) if s[0] != "."]
  for i, s in enumerate(subjects):
    print("Splicing {}/{}".format(i+1, len(subjects)))
    spliceAudio(s, aligned = False)
    spliceAudio(s, aligned = True)



# function: alignExamples
# ---
# Outputs CSV with transcript aligned and leading silence trimmed examples
# Output: subject, start, end, label, interval_number, dialogue
#
# Greedily discards silences and advances start of example
# When overlap occurs, extends example to consume current utterance
#     then sets the start of the next example for the prev. utterance end
#     almost always positive.
def alignExamples(subject):
  output = []
  csv_file = os.path.join(PROCESSED_DATA_PATH, subject, PEDAL_CORR_CSV.format(subject))
  punc_file = os.path.join(RAW_DATA_PATH, subject, "{}_R_16k.punc.TextGrid".format(subject))
  with open(csv_file) as csv, open(punc_file) as punc:
    for _ in range(PUNC_TEXTGRID_HEADER):
      next(punc)

    _, start, end, label, interval = csv.readline().strip(" \n").split(',')
    start, end = float(start), float(end)
    dialogue = ""
    trimming = True

    for s in punc:
      utterance_start, utterance_end = [float(x) for x in s.strip()[2:].split(" ")]
      utterance = next(punc)[1:-1]

      if utterance_end <= end:
        if utterance == "<SIL>" and trimming:
          start = utterance_end
        else:
          trimming = False
          dialogue += utterance
        continue

      if start <= utterance_start  and end <= utterance_end:
        if utterance != "<SIL>":
          end = utterance_end
          dialogue += utterance
        output.append((subject, str(start), str(end), label, interval, dialogue))
        next_csv = csv.readline()
        if next_csv:
          _, start, end, label, interval = next_csv.strip(" \n").split(',')
          start, end = float(start), float(end)
          start = utterance_end
          assert(utterance_start < utterance_end)
          dialogue = ""
          trimming = True
          continue
        else:
          return output

      # Fall through
      print("Error: Problematic Alignment")
      print(start, end)
      print(utterance_start, utterance_end)
      print(subject, interval)
      print(utterance)
      print(dialogue)

def writeAlignedCSV(output, subject):
  # write results, no transcript
  rows = [','.join(o[0:-1]) for o in output]
  rows = '\n'.join(rows)
  outfile = os.path.join(PROCESSED_DATA_PATH, subject, PUNC_ALIGNED_CSV.format(subject))
  with open(outfile, 'w') as outfile:
    outfile.write(rows)

def writeAlignedJSON(output, subject):
  serialized = []
  for o in output:
    example = dict()
    example['subject'] = o[0]
    example['start'] = float(o[1])
    example['end'] = float(o[2])
    example['label'] = o[3]
    example['interval'] = o[4]
    example['transcript'] = o[5]
    serialized.append(example)
  outfile = os.path.join(PROCESSED_DATA_PATH, subject, PUNC_ALIGNED_JSON.format(subject))
  with open(outfile, 'w') as outfile:
    json.dump(serialized, outfile, ensure_ascii=True)

def createAlignedCSV():
  subjects = [s for s in os.listdir(PROCESSED_DATA_PATH) if s[0] != "."]
  for s in subjects:
    output = alignExamples(s)
    writeAlignedCSV(output, s)
    writeAlignedJSON(output, s)
    
  
# function: parsePedalTextGrid
# ---
# Outputs CSV of pedal_hand_corr for each subject
# Output: subject, start, end, label, interval_number
#
def parsePedalTextGrid(subject):
  infile = os.path.join(RAW_DATA_PATH, subject, "{}_pedal_hand_corr.TextGrid".format(subject))
  
  outdir = os.path.join(PROCESSED_DATA_PATH, subject)
  if not os.path.isdir(outdir):
    os.mkdir(outdir)
  
  outfile = os.path.join(outdir, "{}_labels.csv".format(subject))
  
  output = []
  with open(infile, 'r') as inf:
    for _ in range(PEDAL_CORR_HEADER):
      next(inf)
    for line in inf:
      interval = line.strip()[11:][0:-2].strip(" \n")
      lines = [next(inf).strip() for _ in range (3)]
      xmin = float(lines[0][7:])
      xmax = float(lines[1][7:])
      label_text = lines[2][8:][0:-1].lower()
      if label_text:
        # subject, start, end, label, interval_number
        output.append("{},{},{},{},{}".format(subject, xmin, xmax, label_text, interval))
  output = '\n'.join(output)
  with open(outfile, 'w') as outfile:
    outfile.write(output)


def createPedalCSV():
  subjects = [s for s in os.listdir(RAW_DATA_PATH) if s[0] != "."]
  for s in subjects:
    parsePedalTextGrid(s)


def parseArgs():
  parser = argparse.ArgumentParser(description="Data cleaning script TextGrid -> CSV")
  parser.add_argument('--pedal', action='store_true', help='parse pedal_hand_corr files', default = False)
  parser.add_argument('--aligned', action='store_true', help='', default = False)
  parser.add_argument('--audio', action='store_true', help='', default = False)
  parser.add_argument('--silence', action='store_true', help='', default = False)
  parser.add_argument('--all', action='store_true', help='', default = False)
  return parser.parse_args()

def main():
  args = parseArgs()
  if args.pedal or args.all:
    createPedalCSV()
    print("Done pedal csv.")
  if args.aligned or args.all:
    createAlignedCSV()
    print("Done aligned csv and json.")
  if args.audio or args.all:
    trimExamples()
    print("Done trimming audio.")
  if args.silence or args.all:
    removeArtifacts()
    print("Done removing silence and artifacts.")
    
    

if __name__ == '__main__':
  main()
import os
from pathlib import Path
import sys
sys.path.append(Path(os.getcwd()).as_posix())

from src.hsi_moss.moss2 import *

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pathlib import Path
import numpy as np
import math


clf = make_pipeline(StandardScaler(), SVC())

basepath =Path(r"I:\moss_data\Austin moss 2023\Moss\pipeline")
pipeline = MossProcessor(
    basepath,
    basepath
)

X = np.zeros((384,204))
Y = []

csvkeys = MossDatasetCSVKeys()
for idx, path in enumerate(pipeline.stiffs):
    sp_mean_path = DatasetOutput(path.stem, DatasetOutputTypes.specimen_mean, path.parent.parent).filepath
    specimen_mean = np.loadtxt(sp_mean_path.as_posix(), delimiter=',', skiprows=1)
    X[idx,:] = specimen_mean[:,1]
    Y.append(pipeline.csvdata.get(csvkeys.site)[idx])

s1 = [0,96*1-1]
s2 = [96*1,96*2-1]
s3 = [96*2,96*3-1]
s4 = [96*3,96*4-1]

def split(data):
    l = len(data)
    ltest = math.floor(l/4)
    return (data[:ltest],data[ltest:])

trainX, testX = split(X[s1[0]:s1[1]])
trainY, testY = split(Y[s1[0]:s1[1]])
clf.fit(trainX, trainY)


for idx, sd in enumerate(testX):
    print(f'{pipeline.stiffs[idx].name}:\n\tActual: {pipeline.csvdata.get(csvkeys.site)[idx]}\n\tPredicted: {clf.predict([sd])[0]}')

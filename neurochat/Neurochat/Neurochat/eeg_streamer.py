import time
from eeg_engagement import make_synthetic_eeg
import random

FS = 256

def run_stream(scorer):
    print("[EEG] Stream started")   
    while True:
        #this is simulation data, real data goes here
        level = random.choice(["low", "medium", "high"])
        chunk = make_synthetic_eeg(1, FS, 8, level)

        scorer.update(chunk)
        time.sleep(1)
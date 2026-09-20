"""Small demonstration of representation fusion.

For each document we combine a text embedding with a synthetic waveform embedding.
The current baseline uses topic-specific synthetic waveforms, so this is a learning
exercise rather than a scientific seismic classifier.
"""
import numpy as np, pandas as pd
from models.synthetic_waveforms import make_waveform

def main():
    docs=pd.read_csv("artifacts/geophysics_corpus.csv")
    # A simple signal fingerprint baseline: dominant frequency from FFT.
    reps=[]
    for _,r in docs.iterrows():
        x=make_waveform(r.topic,seed=int(r.doc_id[-4:]))
        spec=np.abs(np.fft.rfft(x))
        reps.append(np.argmax(spec[1:])+1)
    docs["waveform_peak_bin"]=reps
    print(docs.groupby("topic")["waveform_peak_bin"].mean().to_string())
    print("Next research step: replace the handcrafted fingerprint with WaveformEncoder and learn text+waveform fusion end-to-end.")

if __name__ == "__main__": main()

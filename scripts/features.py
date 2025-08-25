import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import hilbert

def _stats(x):
    x=np.asarray(x); m=x.mean(); s=x.std()+1e-9
    return {"mean":m,"std":s,"min":x.min(),"max":x.max(),"p2p":x.max()-x.min(),
            "rms":np.sqrt(np.mean(x**2)),"skew":np.mean(((x-m)/s)**3),
            "kurt":np.mean(((x-m)/s)**4),
            "crest": (np.max(np.abs(x))/(np.sqrt(np.mean(x**2))+1e-9))}

def _bands(x, fs, bands):
    X=np.abs(rfft(x)); f=rfftfreq(len(x),1/fs); tot=X.sum()+1e-9
    out={}
    for lo,hi in zip(bands[:-1], bands[1:]):
        mask=(f>=lo)&(f<hi); out[f"band_{lo}_{hi}"]=float(X[mask].sum()/tot)
    out["dom_freq"]=float(f[np.argmax(X)]) if len(f) else 0.0
    return out

def one_channel_feats(x, fs, bands, use_env=True):
    x=np.asarray(x).flatten()
    d={**_stats(x), **_bands(x,fs,bands)}
    if use_env:
        env=np.abs(hilbert(x))
        d.update({f"env_{k}":v for k,v in _bands(env,fs,bands).items()})
    return d

def current_feats(sig_xyz, fs, bands, use_env=True):
    d={}
    for i,ch in enumerate(["x","y","z"]):
        dd=one_channel_feats(sig_xyz[:,i], fs, bands, use_env)
        d.update({f"cur_{ch}_{k}":v for k,v in dd.items()})
    return d

def vibration_feats(sig, fs, bands, use_env=True):
    dd=one_channel_feats(sig, fs, bands, use_env)
    return {f"vib_{k}":v for k,v in dd.items()}

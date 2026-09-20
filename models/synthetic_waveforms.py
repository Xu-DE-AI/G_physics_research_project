import numpy as np

def make_waveform(topic, n=2048, seed=0):
    rng=np.random.default_rng(seed)
    t=np.linspace(0,1,n)
    base={
        "pressure_diffusion":8,"fault_reactivation":12,"injection_rate":20,"localization":16,
        "roughness":28,"permeability":10,"seismicity":24,"velocity":32
    }[topic]
    f=base*(1+0.05*rng.normal())
    envelope=np.exp(-((t-0.35)/0.16)**2)+0.55*np.exp(-((t-0.68)/0.10)**2)
    x=envelope*np.sin(2*np.pi*f*t)+0.15*rng.normal(size=n)
    return x.astype("float32")

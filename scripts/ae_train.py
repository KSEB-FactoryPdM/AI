import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class AE(nn.Module):
    def __init__(self, in_dim, hidden, latent_dim):
        super().__init__()
        enc=[]; d=in_dim
        for h in hidden: enc += [nn.Linear(d,h), nn.ReLU()]; d=h
        enc += [nn.Linear(d,latent_dim)]
        # decoder 대칭
        dec_layers=[]
        d_dec = latent_dim
        for h in reversed(hidden):
            dec_layers += [nn.Linear(d_dec,h), nn.ReLU()]
            d_dec = h
        dec_layers += [nn.Linear(d_dec, in_dim)]
        self.encoder=nn.Sequential(*enc)
        self.decoder=nn.Sequential(*dec_layers)
    def forward(self,x): z=self.encoder(x); xr=self.decoder(z); return xr,z

def fit_ae(X, cfg, epochs_key="epochs"):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import torch.optim as optim
    import torch.nn as nn

    # 안전 캐스팅
    to_float = lambda v: float(v)
    to_int = lambda v: int(float(v))

    X = np.asarray(X, dtype=np.float32)

    hidden = [to_int(h) for h in cfg["hidden"]]
    latent_dim = to_int(cfg["latent_dim"])
    batch_size = to_int(cfg["batch_size"])
    lr = to_float(cfg["lr"])
    epochs = to_int(cfg.get(epochs_key, cfg["epochs"]))

    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    ae = AE(X.shape[1], hidden, latent_dim)
    opt = optim.Adam(ae.parameters(), lr=lr)
    crit = nn.MSELoss()

    ae.train()
    for _ in range(epochs):
        for (xb,) in dl:
            xr, _ = ae(xb)
            loss = crit(xr, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    ae.eval()
    return ae

def encode(ae, X):
    with torch.no_grad():
        z = ae.encoder(torch.from_numpy(np.asarray(X,dtype=np.float32)))
    return z.numpy()

def recon_error(ae, X):
    with torch.no_grad():
        X=torch.from_numpy(np.asarray(X,dtype=np.float32))
        xr,_=ae(X)
        err=((xr-X)**2).mean(dim=1).numpy()
    return err

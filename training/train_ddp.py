"""Minimal DDP learning scaffold.

Run with:
    torchrun --standalone --nproc_per_node=2 -m training.train_ddp

This intentionally uses a tiny synthetic model so the user can understand process
ranks, DistributedDataParallel, DistributedSampler, gradient synchronization, and
checkpointing before attempting multi-GPU Transformer training.
"""
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler


def main():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)
    rank=dist.get_rank(); world=dist.get_world_size()
    device=torch.device("cuda",rank) if torch.cuda.is_available() else torch.device("cpu")
    x=torch.randn(512,20); y=(x[:,0]+0.5*x[:,1]>0).long()
    ds=TensorDataset(x,y); sampler=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True)
    loader=DataLoader(ds,batch_size=32,sampler=sampler)
    model=DDP(nn.Sequential(nn.Linear(20,32),nn.ReLU(),nn.Linear(32,2)).to(device))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3); loss_fn=nn.CrossEntropyLoss()
    for epoch in range(3):
        sampler.set_epoch(epoch)
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            loss=loss_fn(model(xb),yb); opt.zero_grad(); loss.backward(); opt.step()
        if rank==0: print(f"epoch={epoch+1} world_size={world} complete")
    dist.destroy_process_group()

if __name__=="__main__": main()

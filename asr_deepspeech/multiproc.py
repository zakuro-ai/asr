import torch
import sys
import subprocess

argslist = list(sys.argv)[1:]
if not '--world-size' in argslist:
    argslist.append('--world-size')
    argslist.append(str(torch.cuda.device_count()))

if not '--device-ids' in argslist:  # Manually specified GPU IDs
    device_ids = [int(k) for k in range(torch.cuda.device_count())]
else:
    device_ids = [int(k) for k in argslist[argslist.index('--device-ids') + 1].strip().split(',')]
workers = []
world_size = argslist[argslist.index('--world-size') + 1]
rank0 = int(argslist[argslist.index('--rank') + 1])
for gpu_rank in device_ids:
    argslist[argslist.index('--rank') + 1] = str(int(rank0) + int(gpu_rank))
    argslist_k = argslist + ['--gpu-rank', ' ' + str(gpu_rank)]
    print(argslist_k)
    stdout = None if rank0+gpu_rank == 0 else open("GPU_" + str(rank0+gpu_rank) + ".log", "w")
    p = subprocess.Popen([str(sys.executable)] + argslist_k, stdout=stdout, stderr=stdout)
    workers.append(p)

for p in workers:
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(returncode=p.returncode,
                                            cmd=p.args)

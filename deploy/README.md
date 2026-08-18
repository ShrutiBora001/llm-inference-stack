# Deploy

Two paths. Both run identical code; pick by whether the provider gives you
Docker or a bare SSH box.

## One command (rented box, SSH)

```bash
deploy/remote.sh <PORT> <HOST> all
```

Syncs the repo, vendors the upstream `dattn` finding so the box needs **no git
access and no credentials**, sets up the environment idempotently, runs
everything, and pulls results back.

```bash
deploy/remote.sh 16094 23.127.144.217 smoke     # 5 min: validate the hardware
deploy/remote.sh 16094 23.127.144.217 bench     # layout A/B
LIS_BLOCK=128 deploy/remote.sh 16094 23.127.144.217 sweep
```

## Docker

```bash
docker build -f deploy/Dockerfile -t lis:latest .
docker run --gpus all --rm -v "$PWD/results:/workspace/lis/results" lis:latest all
```

The image builds on NGC PyTorch and creates a venv with
`--system-site-packages`, inheriting the container's CUDA-matched torch rather
than pulling a fresh 2 GB copy. Installing plain `torch` on a GPU image is the
classic way to get a driver mismatch.

## Verbs

| Verb | Time | What |
|---|---|---|
| `validate` | ~1 min | full test suite + anti-decoration contract |
| `smoke` | ~5 min | NCCL correctness, topology, **measured** bandwidth |
| `bench` | ~15 min | contiguous vs striped A/B |
| `sweep` | ~30 min | sequence sweep to 1M; OOM is recorded, not fatal |
| `profile` | ~5 min | nsys timeline |
| `all` | ~1 hr | everything, in order |
| `shell` | — | drop into the container |

## Configuration

Environment variables, so nothing is hardcoded per machine:

```bash
LIS_WORLD_SIZE=4      # GPUs
LIS_SEQ=32768         # sequence length
LIS_LAYOUT=striped    # striped | contiguous
LIS_BLOCK=512         # 128 on 40GB cards for the 1M sweep
LIS_RESULTS=./results
```

## Every path runs preflight first

Because a wrong interconnect silently invalidates every timing that follows:

- driver present, else exit
- enough GPUs for `LIS_WORLD_SIZE`, else exit
- **NVLink check** — warns loudly on all-PCIe rather than producing a dull
  result and letting you find out in the writeup
- torch / CUDA / NCCL versions recorded into the results

## Cost discipline

- **One GPU for kernel work** (~$1/hr). Four cannot make one kernel converge faster.
- Four for distributed, eight only for NVSHMEM.
- `make plan PROFILE=a100-40gb` predicts what fits **before** you rent — free.
- `all` ends by telling you to destroy the instance. Rented storage vanishes the
  moment you stop paying, and results are pulled back before that line prints.

## Deliberately not here

No credentials, tokens, or git remotes touch the box. Code goes up by `rsync`
over the SSH key the provider already has. A private repo would otherwise need a
token pasted into a third-party machine's shell.

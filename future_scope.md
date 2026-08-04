# Future scope

Considered and deliberately not in the current plan. Each entry states what it
would buy and why it is deferred, so the decision can be revisited rather than
rediscovered.

### 1. JAX as a differential oracle

An independent implementation in `jax.shard_map` with `lax.ppermute`, which
expresses the ring in roughly ten lines against our sixty. Two independent
implementations agreeing is stronger correctness evidence than one plus a
reference. Autodiff would supply the backward pass the PyTorch version lacks.
Deferred: maintaining two stacks doubles the surface for the whole project.

### 2. JAX as the primary implementation

Rewrite the stack in JAX for cleaner sharding via GSPMD and free autodiff.
Rejected rather than deferred: vLLM and SGLang are both PyTorch, so this
abandons the serving path that motivates the project. Only viable if the work
pivots from inference serving toward training research, which would be a
different project with a different audience.

### 3. Training and the backward pass

`all_gather` does not propagate gradients, so training needs the gradient-aware
variant and a backward pass through the ring. Opens the far larger training
market and makes the striping result apply to pre-training, where long-context
runs cost real money. Deferred because the inference story must be complete and
fast before a second regime is added.

### 4. FP8 and quantization

Would invalidate the random-tensor methodology entirely: FP8 scaling factors
depend on real activation distributions, so Gaussian noise stops being a valid
stand-in. Requires a calibration set, a different correctness argument, and
per-tensor scale management. Genuinely interesting, but it changes what
"correct" means and deserves its own design.

### 5. Multi-node over InfiniBand

The regime where NVSHMEM's advantage is largest, since GPU-initiated RDMA
avoids host involvement across nodes. Needs a real cluster rather than a rented
single box, which puts it outside the current budget. Worth revisiting if
cluster access appears, because it is where the ring topology matters most.

### 6. Speculative decoding

Interacts with ring attention through variable-length verification batches,
which break the even-division assumption the current layouts require. Would
force a general uneven-shard implementation — worthwhile in itself, but it
compounds two hard problems and should follow rather than accompany the decode
path.

### 7. Mixture-of-experts serving

Expert parallelism composes with sequence parallelism as a distinct mesh axis,
producing a 2D scheduling problem with its own load-imbalance question — expert
routing is data-dependent, so the imbalance is dynamic rather than structural.
Orthogonal to this work and considerably larger.

### 8. Uneven sequence sharding

Current layouts require `S % 2P == 0`. Production batches are ragged. Lifting
this means per-chunk offsets everywhere and a load-balance argument that holds
for arbitrary partitions, which is a genuine generalization of the striped
result rather than an implementation detail.

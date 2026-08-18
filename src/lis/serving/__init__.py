"""Serving integrations. vLLM and SGLang, deliberately both.

They overlap heavily, so carrying both is only justified if they are compared on
a workload where their designs actually differ:

  vLLM    PagedAttention + mature continuous batching
          -> target: long context, few requests
  SGLang  RadixAttention prefix caching
          -> target: many short requests with shared prefixes

The finding this phase should produce is not "we integrated two frameworks" but
the crossover point: at what prefix-sharing ratio does SGLang overtake vLLM.

Both integrations consume `lis.kernels.flash_attention`, so each can be
developed and CI-tested against the unfused oracle on a laptop and then run on a
fused kernel on a GPU with no change.
"""

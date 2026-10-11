"""VN v6: two-level compression of the per-neuron parameter block.

The question this stage answers is narrow and empirical: when a population of
neurons is trained on a task, how much of its parameter block is *redundant*,
and can that redundancy be removed in two separate steps -- exact duplicates
fused outright, merely-similar neurons folded into a shared base plus a small
per-neuron modulation -- without giving up quality?

The model is deliberately built so that the first step is provably lossless:
the gate is unnormalised and linear, and the pooled activity is a sum over
neurons, so the whole forward pass is *linear in one neuron's own block*.
Two neurons carrying identical blocks therefore contribute exactly twice what
one carries, and replacing both by a single neuron holding their sum cannot
change a single logit.  That is checked by test, not asserted in prose.
"""

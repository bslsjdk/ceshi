"""Concept wave v1.

The earlier stages (6A / 6D / 6E) all propagated a *scalar* energy.  That is
diffusion: every node moves, nothing is directional, and interference can only
happen through the sign of a real weight.  This package replaces it with the
thing that was asked for in the first place -- a wave over a *concept* field.

Three differences that matter, and each one is measurable:

1.  ``concept`` is a vector per node.  Propagation amplitude is
    ``cos(concept_i, concept_j)``, so energy moves along semantic affinity and
    not along whoever happens to be adjacent.
2.  ``psi`` is complex.  Amplitude *and* phase are carried, so two paths that
    reach the same node can cancel.  Cancellation is the error-correction
    mechanism; there is no explicit forward/backward sweep.
3.  ``mode`` is chosen by concept match, not by an energy threshold.  A node
    whose concept does not match the current context is dormant and is skipped
    entirely -- that is where sparsity comes from.

The wave is run as a damped power iteration: repeat (propagate, interfere,
decay) until the pattern stops changing.  What survives is by construction an
eigenmode of the input-conditioned kernel, which is the "stable mode = current
understanding" of the spec.
"""

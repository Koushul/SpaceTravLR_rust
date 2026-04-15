Ligand–receptor interactions
=============================

SpaceTravLR incorporates **ligand–receptor signaling** into the spatial GRN as grouped modulators alongside other regulatory evidence. Pair priors can come from bundled or custom databases, configuration in ``spaceship_config.toml`` (``[grn]``), and CLI additions.

Practical levers
----------------

* ``[grn].max_ligands`` or ``--max-ligands`` — retain high-expression ligands when filtering DB-derived pairs.
* ``--extra-lr`` — merge additional ``L→R`` pairs in CLI-friendly forms (see ``spacetravlr --help``).
* Training exports (``*_betadata.feather`` and summaries) reflect the effective modulator set for each target gene.

The spatial viewer may link here for in-app help; keep this page stable for deep links.

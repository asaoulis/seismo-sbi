"""AxiSEM ensemble producer: the perturbed 1-D Earth models a database ensemble is built from.

``model_io`` reads and writes the external ``background_model.bm`` format, ``perturb`` draws one
perturbed model from a reference, and ``perturbed_models`` writes a whole ensemble to disk. The
databases themselves are meshed and solved outside this repository.
"""

"""Instaseis backend: seismograms read from precomputed AxiSEM waveform databases.

``querier`` wraps one database, ``simulator`` serves a point source from it, ``ensemble`` draws
from a set of databases to carry theory error, and ``multi_model`` gives disjoint receiver
regions their own ensemble.
"""

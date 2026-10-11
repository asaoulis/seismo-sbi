"""Moment-tensor physics: conventions, decompositions, comparisons and lune angles.

``conventions`` holds the component order, the 3x3 matrix and the two scalar moments.
``decomposition`` gives the moment magnitude, CLVD ratio, north-east-down pyrocko tensor and
nodal planes. ``comparison`` gives the up-south-east pyrocko tensor (``pyrocko_mt``, and
``from_pyrocko`` back), Kagan angles and principal axes. ``lune_angles`` gives the Tape and Tape
source-type angles, and ``quakeml`` converts the moment tensor and hypocentre to and from ObsPy's
``Tensor`` and ``Origin``.
"""

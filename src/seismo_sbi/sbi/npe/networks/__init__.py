"""The embedding net of the NPE.

``seismogram_transformer`` encodes each station's traces with a station encoder
(``station_encoders``, ``cnn_feature_extractor``), mixes stations with the axial transformer
(``axial_transformer``, ``pma_pooling``, ``fused_attention``), and adds the station-geometry and
amplitude embeddings (``positional_encoding``, ``fourier_features``, ``amplitude_embedding``).
"""

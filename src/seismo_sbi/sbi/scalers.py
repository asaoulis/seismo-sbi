
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from seismo_sbi.sbi.configuration import ModelParameters


class SymmetricLogScaler:
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def transform(self, X):
        sign = np.sign(X)
        X_abs = np.abs(X)
        X_clipped = np.clip(X_abs, self.lower_bound, self.upper_bound)
        X_log = np.log10(X_clipped)
        X_scaled = sign * (X_log - np.log10(self.lower_bound)) / (np.log10(self.upper_bound) - np.log10(self.lower_bound))
        X_scaled = X_scaled * 0.5 + 0.5
        return X_scaled

    def inverse_transform(self, X_scaled):
        X_scaled = (X_scaled - 0.5 ) / 0.5
        sign = np.sign(X_scaled)
        X_abs_scaled = np.abs(X_scaled)
        X_unscaled = np.power(10, (X_abs_scaled * (np.log10(self.upper_bound) - np.log10(self.lower_bound))) + np.log10(self.lower_bound))
        X_clipped = sign* np.clip(X_unscaled, self.lower_bound, self.upper_bound)
        return X_clipped
    
class LinearSymmetricLogScaler:
    def __init__(self, lower_bound, upper_bound, linear_range = 0.05):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.linear_range = linear_range

    def transform(self, X):
        sign = np.sign(X)
        X_abs = np.abs(X)
        X_clipped = np.clip(X_abs, 0, self.upper_bound)
        
        X_linear = np.where(X_clipped <= self.lower_bound, 
                            (X_clipped / self.lower_bound) * self.linear_range, 
                            (0.5 - self.linear_range) * (np.log10(X_clipped) - np.log10(self.lower_bound)) / (np.log10(self.upper_bound) - np.log10(self.lower_bound)))
        
        X_scaled = sign * X_linear + 0.5
        return X_scaled

    def inverse_transform(self, X_scaled):
        X_scaled = (X_scaled - 0.5)
        sign = np.sign(X_scaled)
        X_abs_scaled = np.abs(X_scaled)
        # print(np.max((((X_abs_scaled / (0.5 - self.linear_range) )* (np.log10(self.upper_bound) - np.log10(self.lower_bound))) + np.log10(self.lower_bound))))
        
        X_inverse = np.where(X_abs_scaled <= self.linear_range, 
                             (X_abs_scaled / self.linear_range) * self.lower_bound, 
                             10 ** (((X_abs_scaled / (0.5 - self.linear_range) )* (np.log10(self.upper_bound) - np.log10(self.lower_bound))) + np.log10(self.lower_bound)))

        X_unscaled = sign * np.clip(X_inverse, 0, self.upper_bound)
        return X_unscaled


class ZeroOneScaler:
    def __init__(self, bounds):
        self.bounds = bounds
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]
        self.range = self.upper_bound - self.lower_bound

    def transform(self, X):
        X_scaled = (X - self.lower_bound) / self.range
        return X_scaled

    def inverse_transform(self, X_scaled):
        X = X_scaled * self.range + self.lower_bound
        return X

class MomentTensorScaler:
    """Scale/shape reparametrisation of the 6-component moment tensor for NDE training.

    The catalogue Gutenberg-Richter prior spans the scalar moment M0 over many orders
    of magnitude, so a plain min-max (:class:`ZeroOneScaler`) collapses small-magnitude
    events into a vanishing region of scaled space — a near-singular density the flow
    cannot model well.  This scaler instead separates a *log-scaled magnitude* from the
    *scale-invariant orientation*, packed back into a dimension-preserving 6-vector so it
    drops into :class:`FlexibleScaler` unchanged:

        u      = (log10 M0 - log10 M0_min) / (log10 M0_max - log10 M0_min)  in [0, 1]
        m_hat  = m6 / ||m6||                                                (unit, on S^5)
        scaled = 0.5 * (u * m_hat + 1)                                      in [0, 1]^6

    Magnitude is encoded as the radius ``||2*scaled - 1|| = u`` and orientation as its
    direction, so the map is invertible (a.e.) with no dropped components or sign loss.
    ``M0 = ||m6|| / sqrt(2)`` is the library scalar-moment convention
    (``wrapper._scalar_moment``); the inverse rebuilds ``m6 = sqrt(2) * M0 * m_hat``.

    Parameters
    ----------
    bounds:
        ``[lower6, upper6]`` raw-component box (N.m). The largest representable component
        sets ``M0_max = max(|bounds|) / sqrt(2)`` (consistent with the recommended
        ``bounds = +/- sqrt(2) * M0_max``).
    n_decades:
        Magnitude dynamic range in log10 decades: ``log10 M0_min = log10 M0_max -
        n_decades`` (used only when ``log10_m0_range`` is None). Must exceed the prior's
        span ``1.5 * (mw_max - mw_min)`` so that all sampled tensors map to ``u in [0, 1]``.
        NB the window's upper edge comes from ``bounds`` (not ``mw_max``), so the true
        minimum is ``log10 M0_max(bounds) - log10 M0_min(prior)`` — slightly larger than
        the prior span. Prefer ``log10_m0_range`` (``mt_log_decades: auto``) to avoid this.
    log10_m0_range:
        Optional ``(log10 M0_min, log10 M0_max)`` window, overriding ``bounds``/``n_decades``.
        Set from the GR prior's ``[mw_min, mw_max]`` (via ``build_flexible_scaler`` with
        ``mt_log_decades: auto``) so the sampled magnitude maps to ``u in [0, 1]`` exactly
        — the prior limits land on the scaled-space limits, no clipping, no wasted range.
    """

    _SQRT2 = np.sqrt(2.0)

    def __init__(self, bounds=None, n_decades: float = 9.0, log10_m0_range=None):
        if log10_m0_range is not None:
            # Explicit magnitude window (e.g. derived from the GR prior's
            # [mw_min, mw_max] so the sampled range maps to u in [0, 1] exactly).
            lo, hi = float(log10_m0_range[0]), float(log10_m0_range[1])
            if hi <= lo:
                raise ValueError(
                    f"log10_m0_range must be increasing, got ({lo}, {hi})"
                )
            self.log10_m0_min, self.log10_m0_max = lo, hi
        elif bounds is not None:
            bounds = np.asarray(bounds, dtype=float)
            max_abs = float(np.max(np.abs(bounds)))
            if max_abs <= 0:
                raise ValueError("MomentTensorScaler needs non-degenerate moment_tensor bounds")
            self.log10_m0_max = np.log10(max_abs / self._SQRT2)
            self.log10_m0_min = self.log10_m0_max - float(n_decades)
        else:
            raise ValueError("MomentTensorScaler needs either bounds or log10_m0_range")
        self._log_range = self.log10_m0_max - self.log10_m0_min

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        r = np.linalg.norm(X, axis=1, keepdims=True)              # ||m6|| = sqrt(2)*M0
        m0 = r / self._SQRT2
        with np.errstate(divide="ignore"):
            log10_m0 = np.log10(np.where(m0 > 0, m0, 1.0))
        u = (log10_m0 - self.log10_m0_min) / self._log_range
        u = np.clip(u, 0.0, 1.0)                                  # guarantee [0,1] support
        safe_r = np.where(r > 0, r, 1.0)
        m_hat = X / safe_r
        return 0.5 * (u * m_hat + 1.0)

    def inverse_transform(self, X_scaled):
        X_scaled = np.asarray(X_scaled, dtype=float)
        w = 2.0 * X_scaled - 1.0
        u = np.linalg.norm(w, axis=1, keepdims=True)              # = u (>= 0)
        safe_u = np.where(u > 0, u, 1.0)
        m_hat = w / safe_u
        log10_m0 = u * self._log_range + self.log10_m0_min
        m0 = np.power(10.0, log10_m0)
        r = self._SQRT2 * m0
        return r * m_hat


class FlexibleScaler:

    def __init__(self, parameters : ModelParameters, moment_tensor_scaling: str = "linear",
                 mt_log_decades: float = 9.0, mt_log10_m0_range=None):
        """Per-parameter-block scaler into [0, 1].

        ``moment_tensor_scaling`` selects how the moment-tensor block is scaled:
        ``"linear"`` (default) uses the plain :class:`ZeroOneScaler` min-max;
        ``"scale_shape"`` uses :class:`MomentTensorScaler` (log-magnitude +
        scale-invariant unit tensor), better-behaved when M0 spans many orders of
        magnitude. ``mt_log_decades`` is forwarded to :class:`MomentTensorScaler`.
        The training-time and inference-time scalers MUST use the same setting (build
        both with :func:`build_flexible_scaler` from the same config).
        """
        if moment_tensor_scaling not in ("linear", "scale_shape"):
            raise ValueError(
                f"moment_tensor_scaling must be 'linear' or 'scale_shape', got {moment_tensor_scaling!r}"
            )
        self.moment_tensor_scaling = moment_tensor_scaling
        self.indices = []
        self.scalers = []
        self.index_to_param_type = {}
        index = 0
        self.n_features_in_ = parameters.parameter_to_vector('theta_fiducial').shape[0]

        for param_type, params in parameters.theta_fiducial.items():
            if param_type == "moment_tensor" and moment_tensor_scaling == "scale_shape":
                scaler = MomentTensorScaler(
                    np.array(parameters.bounds["moment_tensor"]),
                    n_decades=(mt_log_decades if mt_log10_m0_range is None else 9.0),
                    log10_m0_range=mt_log10_m0_range,
                )
            else:
                scaler = ZeroOneScaler(np.array(parameters.bounds[param_type]))
            self.scalers.append(scaler )
            self.indices.append((index, index + len(params)))
            for i in range(len(params)):
                self.index_to_param_type[index + i] = param_type
            index += len(params)


    def transform(self, X):
        X_scaled = np.zeros_like(X)
        for (start, end), scaler in zip(self.indices, self.scalers):
            X_scaled[:, start:end] = scaler.transform(X[:, start:end])
        
        return X_scaled
    
    def inverse_transform(self, X_scaled):
        X = np.zeros_like(X_scaled)
        for (start, end), scaler in zip(self.indices, self.scalers):
            X[:, start:end] = scaler.inverse_transform(X_scaled[:, start:end])

        return X


def scaler_provenance(scaler) -> dict:
    """A JSON-able fingerprint of the theta scaling ACTUALLY in force.

    Recorded into ``model_meta.json`` at training time and re-derived at inference, so a
    checkpoint carries the scaling it was trained under.  Without this the inverse transform
    is rebuilt from whatever YAML happens to be passed at inference time: edit
    ``ml_scaler``/``bounds`` after training and every recovered moment is silently wrong,
    with no error anywhere.  Compare with :func:`check_scaler_provenance`.

    Captures the resolved NUMBERS (the log10 M0 window), not the config spelling, so
    ``mt_log_decades: auto`` and the equivalent explicit value compare equal — what matters
    is the map, not how it was written.
    """
    out = {"moment_tensor": "linear"}
    mt = getattr(scaler, "mt_scaler", None) or getattr(scaler, "_mt_scaler", None)
    if mt is None:
        # A real FlexibleScaler keeps its per-block sub-scalers in the LIST `self.scalers`,
        # so scanning vars() alone never matches (the list is not a MomentTensorScaler) and
        # every checkpoint would record "linear" — the exact silent failure this function
        # exists to prevent. Search the list first, then fall back to plain attributes for
        # any other scaler shape.
        for candidate in list(getattr(scaler, "scalers", None) or []) + list(vars(scaler).values()):
            if isinstance(candidate, MomentTensorScaler):
                mt = candidate
                break
    if isinstance(mt, MomentTensorScaler):
        out = {"moment_tensor": "scale_shape",
               "log10_m0_min": round(float(mt.log10_m0_min), 9),
               "log10_m0_max": round(float(mt.log10_m0_max), 9)}
    return out


def check_scaler_provenance(meta: dict, scaler, *, strict: bool = False) -> bool:
    """Compare a checkpoint's recorded theta scaling against the one about to be used.

    ``meta`` is the parsed ``model_meta.json``.  Returns True when they agree (or when the
    checkpoint predates the record and nothing can be checked).  A mismatch is the failure
    mode that motivated this: it produces a silent, constant magnitude offset rather than a
    crash, so it is reported loudly and — with ``strict`` — fatally.
    """
    recorded = (meta or {}).get("theta_scaler") or \
        ((meta or {}).get("model_config") or {}).get("theta_scaler")
    if not recorded:
        print("WARNING: checkpoint records no theta_scaler provenance (trained before it was "
              "added); the scaling being used cannot be verified against training.")
        return True
    current = scaler_provenance(scaler)
    same = (recorded.get("moment_tensor") == current.get("moment_tensor")
            and all(abs(float(recorded.get(k, 0.0)) - float(current.get(k, 0.0))) < 1e-6
                    for k in ("log10_m0_min", "log10_m0_max")
                    if k in recorded or k in current))
    if not same:
        msg = ("theta-scaler MISMATCH between checkpoint and config.\n"
               f"    trained with : {recorded}\n"
               f"    about to use : {current}\n"
               "  Recovered moments will be WRONG by a constant factor. Use the config the "
               "checkpoint was trained with, or retrain.")
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg}")
    return same


def build_flexible_scaler(parameters: ModelParameters, raw_config: dict = None) -> FlexibleScaler:
    """Build a :class:`FlexibleScaler`, honouring an optional top-level ``ml_scaler`` block.

    The same helper must be used at training and at inference so the scaling matches::

        ml_scaler:
          moment_tensor: scale_shape   # or "linear" (default)
          mt_log_decades: 9.0          # optional, MomentTensorScaler dynamic range

    ``raw_config`` is the parsed YAML dict (``SBI_Configuration.raw_config`` or a
    ``yaml.safe_load`` of the config file); ``None`` reproduces the legacy default.
    """
    raw_config = raw_config or {}
    cfg = raw_config.get("ml_scaler") or {}
    mt_scaling = cfg.get("moment_tensor", "linear")
    mt_log_decades = cfg.get("mt_log_decades", 9.0)

    mt_log10_m0_range = None
    if mt_scaling == "scale_shape" and isinstance(mt_log_decades, str):
        if mt_log_decades.lower() != "auto":
            raise ValueError(
                f"ml_scaler.mt_log_decades must be a number or 'auto', got {mt_log_decades!r}"
            )
        # Dynamic: fit the log-M0 window to the GR prior so mw_min/mw_max land on u=0/1.
        mt_log10_m0_range = _mt_log10_m0_range_from_prior(raw_config)

    return FlexibleScaler(
        parameters,
        moment_tensor_scaling=mt_scaling,
        mt_log_decades=mt_log_decades,
        mt_log10_m0_range=mt_log10_m0_range,
    )


def _mt_log10_m0_range_from_prior(raw_config: dict):
    """``[log10 M0_min, log10 M0_max]`` window from the moment_tensor Gutenberg-Richter
    prior, so the sampled magnitude range maps to ``u in [0, 1]`` exactly.

    Uses the SAME magnitude->M0 convention as the sampler
    (:func:`seismo_sbi.priors.gutenberg_richter.magnitude_to_m0`,
    ``M0 = 10**(1.5*Mw + 9.1)``) and honours ``magnitude_conversion``.  Requires a
    ``gutenberg_richter`` moment_tensor sampler (``mt_log_decades: auto``).
    """
    from seismo_sbi.priors.gutenberg_richter import magnitude_to_m0

    smpl = (((raw_config.get("simulations") or {}).get("sampling_method") or {})
            .get("moment_tensor"))
    # In the live pipeline, SBI_Configuration resolves this sampling_method entry from
    # its dict form into a built sampler CALLABLE whose ``.info`` carries the derived
    # log10(M0) window; a pristine (un-parsed) YAML config still holds the dict form.
    if callable(smpl):
        rng = (getattr(smpl, "info", {}) or {}).get("log10_m0_range")
        if rng is None:
            raise ValueError(
                "ml_scaler.mt_log_decades: auto — the resolved moment_tensor sampler "
                "exposes no log10_m0_range (needs a gutenberg_richter sampler)"
            )
        return (float(rng[0]), float(rng[1]))
    smpl = smpl or {}
    if smpl.get("type") != "gutenberg_richter":
        raise ValueError(
            "ml_scaler.mt_log_decades: auto requires a gutenberg_richter moment_tensor "
            "sampler (simulations.sampling_method.moment_tensor.type)"
        )
    mw_min, mw_max = float(smpl["mw_min"]), float(smpl["mw_max"])
    conv = smpl.get("magnitude_conversion", "identity")

    def to_mw(m):
        if conv in (None, "identity"):
            return m
        if isinstance(conv, dict):
            return conv.get("slope", 1.0) * m + conv.get("intercept", 0.0)
        raise ValueError(
            "mt_log_decades: auto supports magnitude_conversion 'identity' or "
            "{slope, intercept}; a callable conversion can't be derived from config"
        )

    lo, hi = sorted((float(np.log10(magnitude_to_m0(to_mw(mw_min)))),
                     float(np.log10(magnitude_to_m0(to_mw(mw_max))))))
    return (lo, hi)


class GeneralScaler:
    def __init__(self, raw_compressed_dataset):
        self.log_scaler = SymmetricLogScaler(5*np.min(np.abs(raw_compressed_dataset), axis=0), np.max(np.abs(raw_compressed_dataset),axis=0))
 
        first_transform = self.log_scaler.transform(raw_compressed_dataset)
        self.scaler = MinMaxScaler()
        self.scaler.fit(first_transform)

    def transform(self, X):
        first_transform = self.log_scaler.transform(X)
        return self.scaler.transform(first_transform)
    
    # def inverse_transform(self, X):
    #     # pad X to length 35
    #     X = np.concatenate([X, np.zeros((X.shape[0], self.num_statistics))], axis=1)
    #     first_transform = self.scaler.inverse_transform(X)
    #     return self.log_scaler.inverse_transform(first_transform)[:, :self.n_features_in_]
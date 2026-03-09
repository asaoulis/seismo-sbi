import numpy as np
from scipy.ndimage import gaussian_filter

###############################################################
# Smooth random field generator
###############################################################

def smooth_frac_field(npts, dz_km, corr_length_km, std_frac, seed=None):
    """
    Generate a Gaussian-filtered fractional perturbation field.
    """
    rng = np.random.default_rng(seed)
    white = rng.normal(size=npts)
    sigma_samples = max(0.5, corr_length_km / dz_km)
    smooth = gaussian_filter(white, sigma=sigma_samples, mode='reflect')
    smooth -= np.mean(smooth)
    smooth /= np.std(smooth) + 1e-16
    smooth *= std_frac
    return smooth


###############################################################
# Brocher (2005) density relation
###############################################################

def brocher_rho(vp):
    """
    Brocher (2005) empirical density relation.
    Returns density in g/cm^3.
    """
    vp = np.asarray(vp)
    rho = (1.6612*vp
          - 0.4721*vp**2
          + 0.0671*vp**3
          - 0.0043*vp**4
          + 0.000106*vp**5)
    return rho


###############################################################
# Main perturbation function (works on CPS-format vmodel)
###############################################################

def perturb_cps_model(vmodel,
                      corr_length_km=5.0,
                      std_vp=0.03,
                      std_vs=0.03,
                      std_thickness=0.03,
                      vp_vs_corr=0.9,
                      seed=None):
    """
    Perturb a CPS-format velocity model using smooth fractional perturbations.

    Parameters
    ----------
    vmodel : ndarray of shape (6, N)
        CPS-format model:
          vmodel[0,:] = thickness (km)
          vmodel[1,:] = Vp (km/s)
          vmodel[2,:] = Vs (km/s)
          vmodel[3,:] = density (g/cm^3)
          vmodel[4,:] = Qp
          vmodel[5,:] = Qs

    Returns
    -------
    perturbed_vmodel : ndarray (same shape)
        New perturbed CPS model.
    """

    # Unpack
    H   = vmodel[0].copy()   # thicknesses
    vp  = vmodel[1].copy()
    vs  = vmodel[2].copy()
    rho = vmodel[3].copy()
    Qp  = vmodel[4].copy()
    Qs  = vmodel[5].copy()

    N = len(H)

    # Convert thickness to top-of-layer depth
    depth = np.concatenate(([0.0], np.cumsum(H)))[:-1]
    dz_km = np.median(np.diff(depth)) if N > 1 else H[0]

    #######################################################################
    # ---- Smooth correlated perturbations for Vp and Vs ------------------
    #######################################################################

    rng = np.random.default_rng(seed)

    shared = smooth_frac_field(N, dz_km, corr_length_km,
                               std_frac=1.0,)
    indep1 = smooth_frac_field(N, dz_km, corr_length_km,
                               std_frac=1.0,)
    indep2 = smooth_frac_field(N, dz_km, corr_length_km,
                               std_frac=1.0,)

    a = np.sqrt(max(0.0, min(1.0, vp_vs_corr)))
    b = np.sqrt(1 - a*a)

    eps_vp = a*shared + b*indep1
    eps_vs = a*shared + b*indep2

    eps_vp = eps_vp / np.std(eps_vp) * std_vp
    eps_vs = eps_vs / np.std(eps_vs) * std_vs

    vp_p = vp * np.exp(eps_vp)
    vs_p = vs * np.exp(eps_vs)

    # Enforce Vs < Vp/sqrt(2)
    max_ratio = 1.0 / np.sqrt(2.0)
    mask = vs_p > max_ratio * vp_p
    vs_p[mask] = vp_p[mask] * (0.99 * max_ratio)

    #######################################################################
    # ---- Smooth perturbations to thicknesses ---------------------------
    #######################################################################

    eps_H = smooth_frac_field(N, dz_km, corr_length_km,
                              std_frac=std_thickness)

    H_p = H * np.exp(eps_H)

    #######################################################################
    # ---- Updated density using Brocher --------------------------------
    #######################################################################

    rho_p = brocher_rho(vp_p)
    rho_p = np.maximum(rho_p, 1.0)   # avoid pathological low density

    #######################################################################
    # ---- Pack CPS model back together ---------------------------------
    #######################################################################

    perturbed = np.vstack([H_p, vp_p, vs_p, rho_p, Qp, Qs])

    return perturbed
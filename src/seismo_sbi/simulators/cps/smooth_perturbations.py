"""Draw perturbed layered velocity models in the CPS format.

The compressional and shear speeds get smooth correlated fractional perturbations with a
correlation length in depth, the layer thicknesses get their own, and the density follows the
perturbed compressional speed through the Brocher (2005) relation.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_frac_field(npts, dz_km, corr_length_km, std_frac):
    """A Gaussian-smoothed fractional perturbation field over ``npts`` layers of ``dz_km``, drawn
    from numpy's global random state like every other prior sampler."""
    white = np.random.normal(size=npts)
    sigma_samples = max(0.5, corr_length_km / dz_km)
    smooth = gaussian_filter(white, sigma=sigma_samples, mode='reflect')
    smooth -= np.mean(smooth)
    smooth /= np.std(smooth) + 1e-16
    smooth *= std_frac
    return smooth



def brocher_rho(vp):
    """Density in g/cm^3 from compressional speed in km/s, after Brocher (2005)."""
    vp = np.asarray(vp)
    rho = (1.6612*vp
          - 0.4721*vp**2
          + 0.0671*vp**3
          - 0.0043*vp**4
          + 0.000106*vp**5)
    return rho



def perturb_cps_model(vmodel,
                      corr_length_km=5.0,
                      std_vp=0.03,
                      std_vs=0.03,
                      std_thickness=0.03,
                      vp_vs_corr=0.9):
    """A perturbed copy of the CPS velocity model ``vmodel``, shaped ``(6, n_layers)``.

    The rows are layer thickness in km, compressional and shear speed in km/s, density in
    g/cm^3, then qp and qs. The speeds get smooth correlated fractional perturbations, the
    shear speed staying below the Poisson bound, the thicknesses their own, and the density
    follows the perturbed compressional speed through Brocher (2005).
    """

    H   = vmodel[0].copy()
    vp  = vmodel[1].copy()
    vs  = vmodel[2].copy()
    Qp  = vmodel[4].copy()
    Qs  = vmodel[5].copy()

    N = len(H)

    depth = np.concatenate(([0.0], np.cumsum(H)))[:-1]
    dz_km = np.median(np.diff(depth)) if N > 1 else H[0]



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

    max_ratio = 1.0 / np.sqrt(2.0)
    mask = vs_p > max_ratio * vp_p
    vs_p[mask] = vp_p[mask] * (0.99 * max_ratio)


    eps_H = smooth_frac_field(N, dz_km, corr_length_km,
                              std_frac=std_thickness)

    H_p = H * np.exp(eps_H)


    rho_p = brocher_rho(vp_p)
    rho_p = np.maximum(rho_p, 1.0)


    perturbed = np.vstack([H_p, vp_p, vs_p, rho_p, Qp, Qs])

    return perturbed
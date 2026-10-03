# Conventions

Every quantity is in SI units unless its name says otherwise (`_km`, `_deg`, `_s`, `_hz`).

| Quantity | Convention |
|---|---|
| Moment tensor | `m6 = [m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]` in N m, in the spherical basis r, θ, φ = up, south, east: the Global CMT and Instaseis convention. From Aki & Richards' north-east-down tensor, `m_rr = M_zz`, `m_tt = M_xx`, `m_pp = M_yy`, `m_rt = M_xz`, `m_rp = -M_yz`, `m_tp = -M_xy`. |
| Scalar moment | `M0 = sqrt(½ Σ_ij M_ij²)` over the full symmetric tensor (Silver & Jordan 1982), each off-diagonal component counted twice; `moment_tensor.conventions.scalar_moment`, equal to pyrocko's. |
| Moment magnitude | `Mw = ⅔ (log10 M0 - 9.1)`, M0 in N m (IASPEI 2013); `moment_tensor.conventions.moment_magnitude`. |
| Source type | Tape & Tape (2012) lune longitude γ (`gamma_deg`) and latitude δ (`delta_deg`) from the eigenvalues: a double couple at (0, 0), the explosion at δ = +90°, the implosion at δ = -90°, the CLVD with eigenvalues (2, -1, -1) at γ = -30° and (-2, 1, 1) at γ = +30°, a tensile crack (3, 1, 1) at (-30°, 60.5°). |
| Source position | Latitude and longitude in degrees, depth in km positive downwards from the catalogue datum. |
| Time | Seconds relative to the origin time unless a name says `utc`. A source time is the centroid of the moment-rate function (CPS: its onset). Synthetics from Instaseis start 60 s before the origin time; CPS synthetics start at it. |
| Horizontal components | Stored as east (`E`) and north (`N`); recorded `1`/`2` channels are rotated to them with the station's orientation. |

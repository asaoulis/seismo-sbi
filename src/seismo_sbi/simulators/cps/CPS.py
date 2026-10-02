"""Green's functions from Computer Programs in Seismology.

Writes the velocity model and the distance file the CPS programs read, runs hprep96, hspec96
and hpulse96 over a list of epicentral distances, and reads the elementary Green's functions
back. ``update_with_Gtensor`` rotates them to a receiver's azimuth and into the moment-tensor
frame, so a simulation is a contraction of the tensor with the six components.
"""
import logging
import subprocess
from obspy import read, Stream
from pathlib import Path
import hashlib
import numpy as np

logger = logging.getLogger(__name__)

TEN = 10

def write_Model96(vel_model, fname):
    '''Write the layered model ``vel_model`` to ``fname`` in the Model96 format CPS reads.'''
    thick = vel_model[0, :]
    vp = vel_model[1, :]
    vs = vel_model[2, :]
    rho = vel_model[3, :]
    qka = vel_model[4, :]
    qmu = vel_model[5, :]

    fid = open(fname, 'w')
    fid.write('MODEL.01\n' +
            'Korean Pennisula model from Kim et al. (2011)\n' +
            'ISOTROPIC\n' +
            'KGS\n' +
            'FLAT EARTH\n' +
            '1-D\n' +
            'CONSTANT VELOCITY\n' +
            'LINE08\n' +
            'LINE09\n' +
            'LINE10\n' +
            'LINE11\n')
    fid.write('H(KM)    VP(KM/S) VS(KM/S) RHO(GM/CC)  QP       QS    ETAP     ETAS   FREFP    FREFS\n')
    for n in range(len(thick)):
        line = '%-8.2f %-8.3f %-8.3f %-8.3f %-8.1f %-8.1f %-8.1f %-8.1f %-8.1f %-8.1f\n' % \
            (thick[n], vp[n], vs[n], rho[n], qka[n], qmu[n], 0, 0, 1, 1)
        fid.write(line)
    fid.flush()
    fid.close()

import uuid


def get_hashcode(dists_in_km, evdp_in_km, vmodel):
    """A random MD5 hash string; the arguments are ignored."""
    random_bytes = uuid.uuid4().bytes
    return hashlib.md5(random_bytes).hexdigest()

def perturb_model(vmodel, kappa):
    '''Perturb the velocity model by adding random noise to it.'''
    perturb = vmodel.copy()
    if kappa < 1.: logger.warning('Warning: kappa is too small to make any perturbation')
    perturb[:3, :] *= np.random.normal(1, kappa/100, (3, vmodel.shape[1]))
    return perturb

def calc_CPS_GFs(dists_in_km, evdp_in_km, vmodel, output='DISP',
                 dt=1, npts=512, t0=0, vred=0, wdir='.', verbose=False, cps_path=None):
    """Run the CPS programs for a source at ``evdp_in_km`` and the distances ``dists_in_km``.

    ``vmodel`` is the layered model with columns thickness, compressional speed, shear speed,
    density, qkappa and qmu. ``dt`` is the sample interval in s, ``npts`` the seismogram
    length, ``t0`` the reference time relative to the origin and ``vred`` the reduction speed
    in km/s that sets where each seismogram starts. The Green's functions are written under
    ``wdir``.
    """
    if cps_path is None:
        raise ValueError("CPS path must be provided dynamically.")

    wdir_path = Path(wdir)
    if verbose: logger.info(' Calculate GFs using CPS programs in %s', wdir)
    with open(wdir_path / 'dfile', 'w') as fp:
        if verbose: logger.info('  - Preparing dfile')
        for dist in dists_in_km:
            fp.write('%.1f %.2f %d %.1f %.1f\n' % (dist, dt, npts, t0, vred))
        fp.close()
    if verbose: logger.info('  - Preparing model96')
    vmodel_fname = wdir_path / 'vel.mod'
    write_Model96(vmodel, vmodel_fname)
    if verbose: logger.info('  - Calculating GFs with CPS programs')
    cmd = f'{cps_path}/hprep96 -M vel.mod -d dfile -HS {evdp_in_km} -HR 0.0 -EQEX -R\n'
    cmd += f'{cps_path}/hspec96 > hspec96.out\n'
    cmd += f'{cps_path}/hpulse96 -{output[0]} -p -l 1 > hpulse96.out\n'
    cmd += f'{cps_path}/f96tosac -B hpulse96.out\n'
    cmd += 'rm -f hpulse96.out hspec96.*'
    out = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, shell=True, cwd=wdir)
    if verbose: logger.info(out.stdout)
    gfstream = Stream()
    for sacf in sorted(wdir_path.glob('B*.sac')):
        tr = read(sacf, format='SAC')[0]
        tr.stats.station = sacf.name[1:4]
        tr.stats.location = sacf.name[4:6]
        gfstream.append(tr)
    gfstream.write(wdir_path / 'GF.mseed', format='MSEED')
    cmd = 'rm -f *.sac'
    out = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, shell=True, cwd=wdir)
    if verbose: logger.info('  - Calculated GF written to %s', wdir_path / 'GF.mseed')

def update_with_Gtensor(objstats, vmodel, delta=None, evdp_in_km=None, filter_params=None,
                         force_calc=True, verbose=True, rootdir='.', return_gf=True, gf_directory=None,
                         cps_path=None):
    """Green's functions for every receiver in ``objstats``, rotated into the moment-tensor
    frame, computing them with CPS first if they are not already stored.
    """
    dists = np.unique(np.round(sorted([s.distance for s in objstats]), 1))
    evdp = evdp_in_km if evdp_in_km is not None else objstats[0].event_depth
    if gf_directory is None:
        hashcode = get_hashcode(dists, evdp, vmodel)
        wdir_path = Path(rootdir) / hashcode
        if not wdir_path.exists(): wdir_path.mkdir(parents=True)
        if not (wdir_path / 'GF.mseed').exists() or force_calc:
            calc_CPS_GFs(
                dists,
                evdp,
                vmodel,
                npts=2 * objstats[0].window,
                wdir=str(wdir_path),
                verbose=verbose,
                output='DISP',
                cps_path=cps_path,
            )

        if verbose: logger.info('  - Reading GF from %s', wdir_path / 'GF.mseed')
        gfstream = read(wdir_path / 'GF.mseed', format='MSEED')
    else:
        wdir_path = Path(gf_directory)
        if not wdir_path.exists():
            raise FileNotFoundError(f"GF directory {gf_directory} does not exist.")
        if verbose: logger.info('  - Reading GF from %s', wdir_path / 'GF.mseed')
        gfstream = read(wdir_path / 'GF.mseed', format='MSEED')
    gfstream_processed = Stream()
    for s in objstats:
        gfid = '%03d' % (np.where(dists == np.round(s.distance, 1))[0][0] + 1)
        gftmp = gfstream.select(station=gfid)

        offset = s.t0 if s.vred <= 0 else (s.t0 + s.distance / s.vred)

        local_filter_params = None
        time_shift = 0.0
        if filter_params:
            local_filter_params = dict(filter_params)
            ts = local_filter_params.pop('time_shift', 0.0)
            if ts is None:
                ts = 0.0
            time_shift = float(ts)

        t1 = gftmp[0].stats.starttime + offset + time_shift
        t2 = t1 + s.window

        if local_filter_params:
            pad = 0.1 * s.window
            t1_ext = t1 - pad
            t2_ext = t2 + pad

            tr = gftmp.copy()
            tr.trim(t1_ext, t2_ext, pad=True, fill_value=0)

            tr.taper(max_percentage=0.05, type='cosine')

            tr.filter(**local_filter_params)

            tr = tr.slice(t1, t2, nearest_sample=False)
            if delta is not None:
                tr = tr.resample(1 / delta)

            gfstream_processed.extend(tr)

        else:
            tr = gftmp.slice(t1, t2, nearest_sample=False)
            if delta is not None:
                tr = tr.resample(1 / delta)
            gfstream_processed.extend(tr)
    try:
        ns = len(objstats)
        nc = 3
        ne = 6
        nt = int(s.window / gfstream_processed[0].stats.delta)
        gfarr = np.array([tr.data[:nt] for tr in gfstream_processed]).reshape((ns, TEN, nt))
    except Exception as ex:
        logger.warning('The length of GF function might need to be longer!')
        raise ex
    gf_tensor = np.zeros((ns, nc, ne, nt))
    phi = np.deg2rad([s.azimuth for s in objstats]).reshape((ns, 1))
    # Rotation to the receiver azimuths, after Minson and Dreger (2008); vertical positive up.
    gf_tensor[:,0,0] =  np.cos(2*phi) * gfarr[:,5]/2 - gfarr[:,0]/6 + gfarr[:,8]/3
    gf_tensor[:,0,1] = -np.cos(2*phi) * gfarr[:,5]/2 - gfarr[:,0]/6 + gfarr[:,8]/3
    gf_tensor[:,0,2] =                                 gfarr[:,0]/3 + gfarr[:,8]/3
    gf_tensor[:,0,3] =  np.sin(2*phi) * gfarr[:,5]
    gf_tensor[:,0,4] =  np.cos(phi)   * gfarr[:,2]
    gf_tensor[:,0,5] =  np.sin(phi)   * gfarr[:,2]
    gf_tensor[:,1,0] =  np.cos(2*phi) * gfarr[:,6]/2 - gfarr[:,1]/6 + gfarr[:,9]/3
    gf_tensor[:,1,1] = -np.cos(2*phi) * gfarr[:,6]/2 - gfarr[:,1]/6 + gfarr[:,9]/3
    gf_tensor[:,1,2] =                                 gfarr[:,1]/3 + gfarr[:,9]/3
    gf_tensor[:,1,3] =  np.sin(2*phi) * gfarr[:,6]
    gf_tensor[:,1,4] =  np.cos(phi)   * gfarr[:,3]
    gf_tensor[:,1,5] =  np.sin(phi)   * gfarr[:,3]
    gf_tensor[:,2,0] =  np.sin(2*phi) * gfarr[:,7]/2
    gf_tensor[:,2,1] = -np.sin(2*phi) * gfarr[:,7]/2
    gf_tensor[:,2,3] = -np.cos(2*phi) * gfarr[:,7]
    gf_tensor[:,2,4] =  np.sin(phi)   * gfarr[:,4]
    gf_tensor[:,2,5] = -np.cos(phi)   * gfarr[:,4]
    # Radial and tangential into north and east, after the CPS manual page B-6.
    baz = np.deg2rad([obj.back_azimuth for obj in objstats]).reshape((ns, 1, 1))
    Ncomp = -gf_tensor[:,1] * np.cos(baz) + gf_tensor[:,2] * np.sin(baz)
    Ecomp = -gf_tensor[:,1] * np.sin(baz) - gf_tensor[:,2] * np.cos(baz)
    gf_tensor[:,2,:,:] = Ncomp
    gf_tensor[:,1,:,:] = Ecomp
    if return_gf:
        return gf_tensor
    else:
        for s, obj in enumerate(objstats): obj.update({'Gtensor':gf_tensor[s]})

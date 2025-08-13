# -*- coding: utf-8 -*-
import json
import subprocess
from obspy import read, Stream
from pathlib import Path
import hashlib
import numpy as np
import json

'''
This module provides a set of functions to calculate Green's functions using
Computer Programs in Seismology (CPS) and to process the calculated Green's
functions for the inversion of moment tensor and depth.
'''

## Absolute path to CPS programs
CPS_RPATH = '/home/alex/work/cps/PROGRAMS.330/bin'
## Conversion from degree to meters
DEG2M = 111.195e3
TEN = 10

def write_Model96(vel_model, fname):
    '''
    This function write a 1D model, including specification of layer thicknesses,
    P-, S-wave velocities and densities, to file in Model96 format (used in CPS).
    '''
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

# def get_hashcode(dists_in_km,evdp_in_km,vmodel):
#     '''
#     Return the unique MD5 hash code of the parameter combination in single-station processing.
#     '''
#     ## make sure the number formats/dtypes in input do not produce different hashcodes
#     tmp = dict(dists=list(np.round(dists_in_km, 1).astype(float)), 
#             evdp=[evdp_in_km*1.], 
#             vmodel=list(np.round(vmodel.flatten(), 1)))
#     return hashlib.md5(json.dumps(tmp).encode('utf-8')).hexdigest()
import uuid


def get_hashcode(dists_in_km, evdp_in_km, vmodel):
    """
    Return a random MD5 hash string, ignoring input values.
    """
    random_bytes = uuid.uuid4().bytes  # generates a random UUID
    return hashlib.md5(random_bytes).hexdigest()

def perturb_model(vmodel, kappa, random_seed=None):
    '''
    Perturb the velocity model by adding random noise to the model.
    '''
    perturb = vmodel.copy()
    # assert kappa >= 0 and kappa <= 100
    if kappa < 1.: print ('Warning: kappa is too small to make any perturbation')
    perturb[1:3, :] *= np.random.normal(1, kappa/100, (2, vmodel.shape[1]))
    # else:
    #     if np.abs(kappa) <1: print('Warning: kappa is too small to make any perturbation')
    #     perturb[1:3, :] *= (1 + kappa/100)
    return perturb

def calc_CPS_GFs(dists_in_km,evdp_in_km,vmodel,output='DISP',
                 dt=0.5,npts=512,t0=0,vred=0,wdir='.',verbose=False):
    """
    Wrapper of CPS programs to calculate Green's functions in a velocity model.
    :param dist_in_km: list of epicentra distances in km
    :param evdp_in_km: event depth in km
    :param vmodel: stratified velocity model where columns are thickness, 
        P-wave velocity, S-wave velocity, density, qkappa, qmu
    :param dt: desided time samping interval
    :param npts: length of seismograms
    :param t0: reference time with respect to origin time
    :param vred: reduction velocity to determine caculated seismograms's starttime
    :param wdir: working directory
    :param verbose: display intermediate messages
    """
    wdir_path = Path(wdir)
    if verbose: print (' Calculate GFs using CPS programs in', wdir)
    ## prepare model96 and dfile before actual calculation
    with open(wdir_path/'dfile', 'w') as fp:
        if verbose: print ('  - Preparing dfile')
        for dist in dists_in_km:
            fp.write('%.1f %.2f %d %.1f %.1f\n' % (dist, dt, npts, t0, vred))
        fp.close()
    ## prepare mod96 velocity model
    if verbose: print ('  - Preparing model96')
    vmodel_fname = wdir_path/'vel.mod'
    write_Model96(vmodel, vmodel_fname)
    ## actual calculation of GFs
    if verbose: print ('  - Calculating GFs with CPS programs')
    cmd = '%s/hprep96 -M vel.mod -d dfile -HS %.1f -HR 0.0 -EQEX -R\n' % (CPS_RPATH, evdp_in_km)
    cmd += '%s/hspec96 > hspec96.out\n' % CPS_RPATH
    cmd += '%s/hpulse96 -%s -p -l 1 > hpulse96.out\n'  % (CPS_RPATH, output[0])
    cmd += '%s/f96tosac -B hpulse96.out\n' % CPS_RPATH
    cmd += 'rm -f hpulse96.out hspec96.*'
    out = subprocess.run(cmd,stdout=subprocess.PIPE, text=True,shell=True,cwd=wdir)
    if verbose: print (out.stdout)
    ## convert GF from SAC files into an MSEED file
    gfstream = Stream()
    for sacf in sorted(wdir_path.glob('B*.sac')):
        tr = read(sacf, format='SAC')[0]
        tr.stats.station = sacf.name[1:4]
        tr.stats.location = sacf.name[4:6]
        gfstream.append(tr)
    gfstream.write(wdir_path/'GF.mseed', format='MSEED')
    cmd = 'rm -f *.sac'
    out = subprocess.run(cmd,stdout=subprocess.PIPE, text=True,shell=True,cwd=wdir)
    if verbose: print ('  - Calculated GF written to',wdir_path/'GF.mseed')

def update_with_Gtensor(objstats,vmodel,delta=None,evdp_in_km=None, filter_params=None,
                         force_calc=True,verbose=True,rootdir='.', return_gf=True, gf_directory=None):
    """
    Get Green's functions calculated by Computer Programs in Seismology (CPS).
    The GFs will be generated on the fly if pre-caculated seismograms doesnt 
    exist on disk. If they are, they will be readed from disk.
    
    The actual information to be used are from the distance fields (in km) and 
    in event depth in the SAC files. 
    
    :param dist_in_km: list of epicentra distances in km
    :param evdp_in_km: event depth in km
    :param vmodel: stratified velocity model where columns are thickness, 
        P-wave velocity, S-wave velocity, density, qkappa, qmu
    :param wdir: working directory
    :param force_calc: True to recalculate the GFs if already exists.
    :param verbose: display intermediate messages
    
    :return gf_str: Obspy stream of calculated Green's functions
    """
    ## unique list of distances to compute GF by CPS
    dists = np.unique(np.round(sorted([s.distance for s in objstats]), 1))
    evdp = evdp_in_km if evdp_in_km is not None else objstats[0].event_depth
    ## generate a unique hashcode for the combination
    if gf_directory is None:
        hashcode = get_hashcode(dists,evdp,vmodel)
        wdir_path = Path(rootdir)/hashcode
        ## caclulate GFs if they haven't been calculated yet or re-calculation forced
        if not wdir_path.exists(): wdir_path.mkdir(parents=True)
        if not (wdir_path/'GF.mseed').exists() or force_calc:
            calc_CPS_GFs(dists,evdp,vmodel,npts=2*objstats[0].window,wdir=str(wdir_path),verbose=verbose,output='DISP')
        
        ## read sac files into an Obspy stream
        if verbose: print ('  - Reading GF from', wdir_path/'GF.mseed')
        gfstream = read(wdir_path/'GF.mseed', format='MSEED')
    else:
        wdir_path = Path(gf_directory)
        if not wdir_path.exists():
            raise FileNotFoundError(f"GF directory {gf_directory} does not exist.")
        if verbose: print ('  - Reading GF from', wdir_path/'GF.mseed')
        gfstream = read(wdir_path/'GF.mseed', format='MSEED')
    gfstream_processed = Stream()
    ## preprocess GF and cut window
    for s in objstats:
        gfid = '%03d'%(np.where(dists==np.round(s.distance,1))[0][0]+1)
        gftmp = gfstream.select(station=gfid)
        ## bandpass filter
        if delta is not None: gftmp.resample(1/delta)
        # gftmp.filter('bandpass', **filter_params)
        start, end = gftmp[0].stats.starttime, gftmp[0].stats.endtime
        # length = (end - start)*delta
        if filter_params:
            # gftmp = gftmp.trim(starttime=start - length * 0.3, endtime=end + length * 0.3, pad = True, fill_value=0)
            gftmp = gftmp.taper(max_percentage=0.02, type='cosine')
            # seismograms = seismograms.filter('bandpass', freqmin=0.04, freqmax=0.07, corners=4, zerophase=False)
            gftmp = gftmp.filter(**filter_params)
        ## window
        offset = s.t0 if s.vred<=0 else (s.t0+s.distance/s.vred)
        t1 = gftmp[0].stats.starttime+offset
        t2 = t1 + s.window
        gfstream_processed.extend(gftmp.slice(t1,t2,nearest_sample=False))
    ## Greens tensor of CPS elementary GFs
    try:
        ns = len(objstats)
        nc = 3 # ZRT
        ne = 6 # mxx, myy, mzz, mxy, mxz, myz
        nt = int(s.window/gfstream_processed[0].stats.delta)
        # print(nt, s.window, gfstream_processed[0].stats.delta, flush=True)
        # print(np.array([tr.data.shape for tr in gfstream_processed]), flush=True)
        gfarr = np.array([tr.data[:nt] for tr in gfstream_processed]).reshape((ns,TEN,nt))
    except Exception as ex:
        print('The length of GF function might need to be longer!')
        ## TODO: fix with with meaningful control.
        raise ex
    ## Greens tensor of 6 oriented GFs
    gf_tensor = np.zeros((ns, nc, ne, nt))
    ## azimuthal angles in radians
    phi = np.deg2rad([s.azimuth for s in objstats]).reshape((ns, 1))
    ## rotate vertical GF to actual azimuths (Minson & Dreger 2008) | POSITIVE UP (FILE96's DIP = -90)
    gf_tensor[:,0,0] =  np.cos(2*phi) * gfarr[:,5]/2 - gfarr[:,0]/6 + gfarr[:,8]/3 # Z.XX
    gf_tensor[:,0,1] = -np.cos(2*phi) * gfarr[:,5]/2 - gfarr[:,0]/6 + gfarr[:,8]/3 # Z.YY
    gf_tensor[:,0,2] =                                 gfarr[:,0]/3 + gfarr[:,8]/3
    gf_tensor[:,0,3] =  np.sin(2*phi) * gfarr[:,5]
    gf_tensor[:,0,4] =  np.cos(phi)   * gfarr[:,2]
    gf_tensor[:,0,5] =  np.sin(phi)   * gfarr[:,2]
    ## rotate radial GF to actual azimuths
    gf_tensor[:,1,0] =  np.cos(2*phi) * gfarr[:,6]/2 - gfarr[:,1]/6 + gfarr[:,9]/3 # R.XX
    gf_tensor[:,1,1] = -np.cos(2*phi) * gfarr[:,6]/2 - gfarr[:,1]/6 + gfarr[:,9]/3
    gf_tensor[:,1,2] =                                 gfarr[:,1]/3 + gfarr[:,9]/3
    gf_tensor[:,1,3] =  np.sin(2*phi) * gfarr[:,6]
    gf_tensor[:,1,4] =  np.cos(phi)   * gfarr[:,3]
    gf_tensor[:,1,5] =  np.sin(phi)   * gfarr[:,3]
    ## rotate tangential GF to actual azimuths
    gf_tensor[:,2,0] =  np.sin(2*phi) * gfarr[:,7]/2                               # T.XX
    gf_tensor[:,2,1] = -np.sin(2*phi) * gfarr[:,7]/2
    gf_tensor[:,2,3] = -np.cos(2*phi) * gfarr[:,7]
    gf_tensor[:,2,4] =  np.sin(phi)   * gfarr[:,4]
    gf_tensor[:,2,5] = -np.cos(phi)   * gfarr[:,4]
    ## rotate radial, tangential components into north, east (page B-6, CPS manual v3.30)
    baz = np.deg2rad([obj.back_azimuth for obj in objstats]).reshape((ns, 1, 1))
    Ncomp = -gf_tensor[:,1] * np.cos(baz) + gf_tensor[:,2] * np.sin(baz)
    Ecomp = -gf_tensor[:,1] * np.sin(baz) - gf_tensor[:,2] * np.cos(baz)
    gf_tensor[:,2,:,:] = Ncomp # Y
    gf_tensor[:,1,:,:] = Ecomp # X
    ## GF tensor in ndarray (for rapid calculation later)
    if return_gf:
        return gf_tensor
    else:
        for s, obj in enumerate(objstats): obj.update({'Gtensor':gf_tensor[s]})

def update_with_dGtensor(dep_array, objstats,vmodel,filter_params,delta=None,
                         force_calc=False,verbose=False,rootdir='.'):
    tmpstats = objstats.copy()
    dGtensor = []
    for evdp in dep_array:
        update_with_Gtensor(tmpstats,vmodel,filter_params,delta=delta,evdp_in_km=evdp,
                             force_calc=force_calc,verbose=verbose,rootdir=rootdir)
        dGtensor.append([obj.Gtensor for obj in tmpstats])
    dGtensor = np.array(dGtensor)
    ## update the stats
    for _s in range(len(objstats)):
        objstats[_s].update({'dGtensor':np.array(dGtensor[:, _s, ...]), 'dep_array':dep_array})
    
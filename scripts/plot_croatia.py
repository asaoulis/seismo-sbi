import sys
import os
import argparse
import shutil
import pickle
from pathlib import Path
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt

from seismo_sbi.sbi.configuration import SBI_Configuration
from seismo_sbi.sbi.pipeline import SingleEventPipeline, SBIPipelinePlotter


from seismo_sbi.instaseis_simulator.receivers import Receivers



rec = Receivers(path_to_stations="./configs/croatia/reduced_geom/event1_stations.txt")

from cartopy import crs as ccrs
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

rec.plot(ax=ax)
plt.savefig("croatia_stations_map.png")
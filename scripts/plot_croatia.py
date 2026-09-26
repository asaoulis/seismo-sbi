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
from seismo_sbi.sbi.pipeline import SingleEventPipeline
from seismo_sbi.plotting.results_plotting import SBIPipelinePlotter


from seismo_sbi.simulators.receivers import Receivers



rec = Receivers(path_to_stations="./configs/croatia/reduced_geom/event1_stations.txt")

from cartopy import crs as ccrs
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

rec.plot(ax=ax)
plt.savefig("croatia_stations_map.png")
"""Display names, units and unit conversions for posterior parameters.

:class:`ParameterInformation` pairs a parameter with its label, unit and a scaling transform.
:class:`DegreeKMConverter` turns latitude and longitude offsets into kilometres.
"""

from typing import NamedTuple, Callable
from enum import Enum

from seismo_sbi.priors.geo import KM_PER_DEG, km_per_degree_longitude

class DegreeType(Enum):
    """Which horizontal coordinate a degree offset is measured along."""

    LATITUDE = 0
    LONGITUDE = 1


class DegreeKMConverter():
    """Kilometres from ``minimum_deg_value`` along ``degree_type``. A degree of longitude is
    ``KM_PER_DEG * cos(reference_latitude_deg)`` long."""

    def __init__(self, minimum_deg_value, degree_type : DegreeType, reference_latitude_deg):
        
        self.minimum_deg_value = minimum_deg_value
        self.scale = km_per_degree_longitude(reference_latitude_deg) if (degree_type is DegreeType.LONGITUDE) else KM_PER_DEG

    def __call__(self, deg_column):
        return (deg_column - self.minimum_deg_value) * self.scale


def no_scaling(x):
    return x

class ParameterInformation(NamedTuple):

    name : str
    unit : str
    scaling_transform : Callable = no_scaling


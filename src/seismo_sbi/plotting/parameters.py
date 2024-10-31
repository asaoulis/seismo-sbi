
from typing import NamedTuple, List, Callable
from enum import Enum

# TODO SHOULD MAKE THIS GENERAL
UPFLOW_LONGITUDE_SCALE = 86
LATITUDE_DEG_TO_KM_SCALE = 111.1

class DegreeType(Enum):

    LATITUDE = 0
    LONGITUDE = 1


class DegreeKMConverter():

    def __init__(self, minimum_deg_value, degree_type : DegreeType):
        
        self.minimum_deg_value = minimum_deg_value
        self.scale = UPFLOW_LONGITUDE_SCALE if (degree_type is DegreeType.LONGITUDE) else LATITUDE_DEG_TO_KM_SCALE

    def __call__(self, deg_column):
        return (deg_column - self.minimum_deg_value) * self.scale


def no_scaling(x):
    return x

class ParameterInformation(NamedTuple):

    name : str
    unit : str
    scaling_transform : Callable = no_scaling


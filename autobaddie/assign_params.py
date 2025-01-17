import os
from munch import Munch
import pandas as pd

"""based on the images provided by the create_template.py file, assign LJ parameters for each atomic environment"""

WORKDIR_BASE = "/home/pleon/projects/AutoBADDIE"  # path to autobaddie home directory
job_name = "autoBADDIE_example"
job_details = Munch()

# glyme.Li.TFSI lennard jones parameters for OPLS parameterization
# litfsi comes from this (Doherty 2017): https://pubs.acs.org/doi/10.1021/acs.jctc.7b00520
custom_parameter_dict = {
    "0": {"types": [0], "sigma": 2.500, "epsilon": 0.030, "charge": None},  # H, CO-()
    "1": {"types": [1], "sigma": 2.500, "epsilon": 0.030, "charge": None},  # H, (CCO)
    "2": {"types": [2], "sigma": 2.13, "epsilon": 0.018, "charge": None},  # Li
    "3": {"types": [3], "sigma": 3.500, "epsilon": 0.066, "charge": None},  # C, CO-()
    "4": {
        "types": [4],
        "sigma": 3.500,
        "epsilon": 0.066,
        "charge": None,
    },  # C, first CO-C*CO(CCO)
    "5": {"types": [5], "sigma": 3.500, "epsilon": 0.066, "charge": None},  # C, (CCO)
    "6": {"types": [6], "sigma": 3.500, "epsilon": 0.066, "charge": None},  # C, TFSI
    "7": {"types": [7], "sigma": 3.250, "epsilon": 0.170, "charge": None},  # N, TFSI
    "8": {
        "types": [8],
        "sigma": 2.900,
        "epsilon": 0.140,
        "charge": None,
    },  # O, CO-(), glyme
    "9": {
        "types": [9],
        "sigma": 2.900,
        "epsilon": 0.140,
        "charge": None,
    },  # O, (CCO), glyme
    "10": {"types": [10], "sigma": 2.960, "epsilon": 0.210, "charge": None},  # O=, TFSI
    "11": {"types": [11], "sigma": 2.950, "epsilon": 0.053, "charge": None},  # F, TFSI
    "12": {"types": [12], "sigma": 3.550, "epsilon": 0.250, "charge": None},  # S, TFSI
}

# ---------------save these figures
job_details.job_name = job_name

job_details.WORKDIR = os.path.join(
    WORKDIR_BASE, "training_results", job_details.job_name, "template"
)

df = pd.DataFrame()
df["custom_type"] = list(custom_parameter_dict.keys())
for key in ["types", "sigma", "epsilon", "charge"]:
    df[key] = df["custom_type"].apply(lambda c: custom_parameter_dict[c][key])
df = df.set_index("custom_type")
path = os.path.join(job_details.WORKDIR, "template.params")
# save the parameters to the training
df.to_csv(path)

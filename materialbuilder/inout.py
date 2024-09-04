import json, os, datetime, pytz
from munch import Munch
import numpy as np
import torch, random
import subprocess, glob, math, copy
    
def read_job_details(path):
    with open(path, 'r') as json_file:
        job_details = json.load(json_file)
    job_details = Munch(job_details)
    return(job_details)

def read_job_output(path, keys):
    if type(keys) != list:
        keys = [keys]
    with open(path) as json_file:
        job_output = json.load(json_file)
    value = getFromDict(job_output, keys)
    return(value)

def populate_default_details(WORKDIR, job_details):
    job_details = {**job_details}
    for key in ['shuffle_geoms', 'use_1_4_pairs']:
        if key not in job_details.keys():
            job_details[key] = True
    if 'species_train_test_split_method' not in job_details.keys():
        job_details['species_train_test_split_method'] = None
    if 'adjmat_source' not in job_details.keys():
        job_details['adjmat_source'] = 'minimum_energy_geometry'
    if 'energy_key' not in job_details.keys():
        job_details['energy_key'] = 'energy'
    if 'custom_types' not in job_details.keys():
        job_details['custom_types'] = []
    if 'terms' not in job_details.keys():
        job_details['terms'] = []
    if 'batch_multiplier' not in job_details.keys():
        job_details['batch_multiplier'] = 1
    if 'random_seed' not in job_details.keys():
        job_details['random_seed'] = None
    if 'properties_to_calculate' not in job_details.keys():
        job_details['properties_to_calculate'] = ['E']
    if 'model_stat_steps' not in job_details.keys():
        b = float(max(job_details['num_train'], job_details['num_test']))
        b = math.ceil(b/job_details['batch_size'])
        job_details['model_stat_steps'] = {'epoch': 1, 'batch': b}
    if 'normalize' not in job_details.keys():
        job_details['normalize'] = []
    if 'subtract_reference_energy' not in job_details.keys():
        job_details['subtract_reference_energy'] = False
    with open(os.path.join(WORKDIR, 'job_details_verbose.json'), 'w') as json_file:
        json.dump(job_details, json_file, indent=4)
    job_details = Munch(job_details)
    return(job_details)

def getFromDict(dic, keys):    
    for k in keys:
        dic = dic[k]
    return(dic)

def setInDict(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

def initialize_job_output(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    with open(path, 'w') as json_file:
        job_output = {}
        job_output['create_time'] = str(datetime.datetime.now(tz=pytz.utc))
        json.dump(job_output, json_file, indent=4)
        
def update_job_output(path, keys, value):
    if type(keys) != list:
        keys = [keys]
    with open(path, 'r') as json_file:
        job_output = json.load(json_file)
    setInDict(job_output, keys, value)
    with open(path, 'w') as json_file:
        json.dump(job_output, json_file, indent=4)
        
def set_random_state(random_seed=None):
    if random_seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        random_state = np.random.RandomState(seed=random_seed)
    else:
        print("No random seed was provided, so not setting random state.")
        
def clean_workdir(WORKDIR):
    paths = []
    paths.append(os.path.join(WORKDIR, 'job_details_verbose.json'))
    paths.append(os.path.join(WORKDIR, 'job_output.json'))
    paths.append(os.path.join(WORKDIR, 'django-debug.log'))
    paths.append(os.path.join(WORKDIR, 'jobdirparser.log'))
    paths.append(os.path.join(WORKDIR, 'loss.csv'))
    paths.append(os.path.join(WORKDIR, 'predictions.csv'))
    paths.append(os.path.join(WORKDIR, 'probe_params.json'))
    paths.append(os.path.join(WORKDIR, 'learned_probe_params.json'))
    paths.append(os.path.join(WORKDIR, 'trained_autopology_FF.json'))
    paths.append(os.path.join(WORKDIR, 'output_lammps_data_file.lmp'))
    paths.append(os.path.join(WORKDIR, 'autopology.data'))
    paths.append(os.path.join(WORKDIR, 'autopology.json'))
    paths.append(os.path.join(WORKDIR, 'data.restart'))
    paths.append(os.path.join(WORKDIR, 'log.lammps'))
    paths.append(os.path.join(WORKDIR, 'pe.xyz'))
    paths.append(os.path.join(WORKDIR, 'trajectory.xyz'))
    paths.append(os.path.join(WORKDIR, 'frames.pickle'))
    paths.append(os.path.join(WORKDIR, 'visualization.png'))
    for path in glob.glob('./[0-9]*[0-9].json'):
        paths.append(path)
    for path in glob.glob('./[0-9]*[0-9].data'):
        paths.append(path)
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

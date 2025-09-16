from six.moves.configparser import ConfigParser
import os
from tqdm import tqdm

config = ConfigParser() # global config variable
config.read('/inwdata2/Prithvi/GIT/analytics/src/nd_config_bagheera3_US.ini') # read the config file

for k, v in tqdm(config.items('deviceModelFiles')): 
    if k.endswith('path'):
        model = v.split('/')[-1]
        if not os.path.exists('/inwdata2/Prithvi/GIT/analytics/models/{}'.format(model)):
            print(model)
            os.system('aws s3 sync s3://netradyne-sharing/analytics/models/{}/ /inwdata2/Prithvi/GIT/analytics/models/{}/ --only-show-errors'.format(model, model))
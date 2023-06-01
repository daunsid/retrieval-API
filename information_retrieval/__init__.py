import configparser
import os
#from dotenv import load_dotenv, find_dotenv


config = configparser.ConfigParser()
config.read('config.ini')

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# load up the entries as environment variables

# a best practice found here:
# https://drivendata.github.io/cookiecutter-data-science/
# find .env automatically by walking up directories until it's found
"""dotenv_path = find_dotenv()
load_dotenv(dotenv_path)"""
'''
YAML file management module
'''
import yaml


def read_yaml_file(filename: str) -> dict:
    '''
    Function to read a YAML file from Disk and return a dictionary.
    '''

    with open(filename, 'r', encoding='utf8') as file:
        yaml_dict = yaml.safe_load(file)

    return yaml_dict

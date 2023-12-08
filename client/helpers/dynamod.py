'''
Module to load modules dynamically from strings.
'''
import traceback
import imp
from helpers.logging import logger


def load_module(module_name: str, module_content: str):
    '''
    Load a module from its string
    '''
    mymodule = imp.new_module(module_name)
    # try:
    exec(module_content, mymodule.__dict__)
    # except Exception:
    #     logger.error(f'Module Execution Failure!\n{traceback.format_exc()}')

    return mymodule

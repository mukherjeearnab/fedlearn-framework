'''
Client Configuration Management module
'''

import socket


def get_system_info():
    '''
    Obtain Base System Info
    '''
    sys_info = {
        'hostname': socket.gethostname(),
    }

    return sys_info

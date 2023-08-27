'''
Commands for Server
'''

from typing import List
from apps.server.controller import start_server, stop_server


def handle_command(args: List[str]) -> None:
    '''
    Handle the Forwarded Commands
    '''
    if args[0] == 'start':
        start_server()

    elif args[0] == 'stop':
        stop()

    elif args[0] == 'help':
        _help()

    else:
        print('Unknown server command: ', args)
        _help()


def stop() -> None:
    '''
    Stop Method to call and stop the server
    '''
    stop_server()


def _help() -> None:
    '''
    Help Method
    '''
    print('''Server Management Help.
Command\tDescription\n
help\tShow this help message.
start\tStart the server.
stop\tStop the server.
''')

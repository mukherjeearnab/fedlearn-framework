'''
Commands for Server
'''

from typing import List
from apps.client_manager import ClientManager

client_manager = ClientManager()


def handle_command(args: List[str]) -> None:
    '''
    Handle the Forwarded Commands
    '''
    if args[0] == 'list':
        client_manager.show_clients()

    elif args[0] == 'help':
        _help()

    else:
        print('Unknown server command: ', args)
        _help()


def _help() -> None:
    '''
    Help Method
    '''
    print('''Server Management Help.
Command\tDescription\n
help\tShow this help message.
list\tList all the clients and their details.
''')

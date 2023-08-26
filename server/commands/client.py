'''
Commands for Server
'''

from typing import List
import json
from apps.client_manager import ClientManager

client_manager = ClientManager()


def handle_command(args: List[str]) -> None:
    '''
    Handle the Forwarded Commands
    '''
    if args[0] == 'list':
        print(json.dumps(client_manager.get_clients(), sort_keys=True, indent=4))

    elif args[0] == 'alive':
        print(json.dumps(client_manager.get_alive_clients(), sort_keys=True, indent=4))

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

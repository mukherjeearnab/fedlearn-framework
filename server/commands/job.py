'''
Commands for Server
'''

from typing import List
from apps.job_loader import load_job, start_job


def handle_command(args: List[str]) -> None:
    '''
    Handle the Forwarded Commands
    '''
    if args[0] == 'load':
        load_job(job_name=args[1])

    elif args[0] == 'start':
        start_job(job_name=args[1])

    elif args[0] == 'help':
        _help()

    else:
        print('Unknown server command: ', args)
        _help()


def _help() -> None:
    '''
    Help Method
    '''
    print('''Job Management Help.
Command\tDescription\n
load [job name]\tLoad the config of a job with name [job name].
start [job name]\tStart the job with [job name].
stop\tStop the server.
''')

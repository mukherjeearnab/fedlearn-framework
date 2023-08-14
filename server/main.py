'''
This is the Commandline interface for managing the server
'''
from commands import server, clients


WELCOME_PROMPT = '''Welcome to the FedLearn Server.
To get started, enter 'help' in the command prompt below.
'''

SINGLE_COMMANDS = ['exit']


if __name__ == '__main__':
    print(WELCOME_PROMPT)
    while True:
        print('> ', end='')
        command = input()
        args = command.split(' ')
        if len(args) <= 1 and args[0] not in SINGLE_COMMANDS:
            continue

        if args[0] == 'server':
            server.handle_command(args[1:])

        elif args[0] == 'client':
            clients.handle_command(args[1:])

        elif args[0] == 'exit':
            server.stop()
            print('Exiting...')
            exit()

        else:
            print('Unknown command: ', command)

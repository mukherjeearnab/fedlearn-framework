'''
This is the Commandline interface for managing the server
'''
from server import start_server, stop_server


WELCOME_PROMPT = '''Welcome to the FedLearn Server.
To get started, enter 'help' in the command prompt below.
'''


if __name__ == '__main__':
    print(WELCOME_PROMPT)
    while True:
        print('> ', end='')
        command = input()

        if command == 'start server':
            start_server()

        elif command == 'exit':
            stop_server()
            print('Exiting...')
            exit()

        else:
            print('Unknown command: ', command)

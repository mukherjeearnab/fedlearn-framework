'''
The Controller Module
'''
import os
from multiprocessing import Process
from dotenv import load_dotenv
from apps.server.root import app

# import environment variables
load_dotenv()

SERVER_PORT = int(os.getenv('SERVER_PORT'))
# print(type(SERVER_PORT))


def run_server():
    '''
    Method to create a thread for the server process
    '''
    app.run(port=SERVER_PORT, debug=False, host='0.0.0.0')


server = Process(target=run_server)


def start_server():
    '''
    Method to start the server
    '''
    server.start()
    print('Server started...')


def stop_server():
    '''
    Method to stop the server, it joins it, and then exit will be called
    '''
    print('Stopping server...')
    if server.is_alive():
        server.terminate()
        server.join()

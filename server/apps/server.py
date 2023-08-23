'''
Main entry point for the server application.
This hosts all the server routes and invokes the main function.
'''
import os
from routes.client_manager import blueprint as client_manager
from routes.job_manager import blueprint as job_manager
from multiprocessing import Process
from flask import Flask, jsonify
from dotenv import load_dotenv


# import environment variables
load_dotenv()

SERVER_PORT = int(os.getenv('SERVER_PORT'))
# print(type(SERVER_PORT))

# import the routers for different routes

app = Flask(__name__)


@app.route('/')
def root():
    '''
    server root route, provides a brief description of the server,
    with some additional information.
    '''
    data = {'message': 'This is the fedlrn-framework server.'}
    return jsonify(data)


# register the blueprint routes
app.register_blueprint(client_manager, url_prefix='/client_manager')
app.register_blueprint(job_manager, url_prefix='/job_manager')


def run_server():
    '''
    Method to create a thread for the server process
    '''
    app.run(port=SERVER_PORT, debug=False, threaded=True)


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

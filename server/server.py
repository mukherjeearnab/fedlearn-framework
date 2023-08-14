'''
Main entry point for the server application.
This hosts all the server routes and invokes the main function.
'''
from multiprocessing import Process
from flask import Flask, jsonify


# import the routers for different routes
from routes.client_manager import blueprint as client_manager

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
app.register_blueprint(client_manager, url_prefix='/client-manager')


def run_server():
    '''
    Method to create a thread for the server process
    '''
    app.run(debug=False, threaded=True)


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
    server.terminate()
    server.join()

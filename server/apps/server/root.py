'''
Main entry point for the server application.
This hosts all the server routes and invokes the main function.
'''
import os
from flask import Flask, jsonify
from dotenv import load_dotenv
from routes.client_manager import blueprint as client_manager
from routes.job_manager import blueprint as job_manager


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

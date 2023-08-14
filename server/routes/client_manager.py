'''
Client Management Routing Module
'''

from flask import Blueprint, jsonify, request
from threading import Semaphore
from apps.client_manager import ClientManager


ROUTE_NAME = 'client-manager'
blueprint = Blueprint(ROUTE_NAME, __name__)

REGISTER_LOCK = Semaphore()
client_manager = ClientManager()


@blueprint.route('/')
def root():
    '''
    blueprint root route, provides a brief description of the route,
    with some additional information.
    '''
    data = {'message': f'This is the \'{ROUTE_NAME}\' router.'}
    return jsonify(data)


@blueprint.route('/register', methods=['POST'])
def register():
    '''
    register route, for clients to register on the server and obtain IDs.
    '''
    REGISTER_LOCK.acquire()

    data = {'id': client_manager.register_client(
        request.get_json(), request.remote_addr)}

    REGISTER_LOCK.release()
    return jsonify(data)

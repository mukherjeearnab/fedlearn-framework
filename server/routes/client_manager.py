'''
Client Management Routing Module
'''

from threading import Semaphore
from flask import Blueprint, jsonify, request
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


@blueprint.route('/get')
def getall():
    '''
    route to get all registered clients
    '''
    REGISTER_LOCK.acquire()

    data = client_manager.get_clients()

    REGISTER_LOCK.release()

    return jsonify(data)


@blueprint.route('/get_alive')
def getalive():
    '''
    route to get alive clients
    '''
    REGISTER_LOCK.acquire()

    data = client_manager.get_alive_clients()

    REGISTER_LOCK.release()

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

    if data['id'] is None:
        return jsonify({'message': 'Registration is Locked!'}), 403

    return jsonify(data)


@blueprint.route('/ping', methods=['POST'])
def ping():
    '''
    register route, for clients to register on the server and obtain IDs.
    '''
    REGISTER_LOCK.acquire()

    req = request.get_json()

    resp = client_manager.alive_ping(req['client_id'])

    REGISTER_LOCK.release()

    if not resp:
        return jsonify({'message': 'Client is not register or NOT Found.'}), 404

    return jsonify({'message': 'Client ping updated.'})

'''
Client Management Routing Module
'''

from flask import Blueprint, jsonify, request
from threading import Semaphore


ROUTE_NAME = 'client-manager'
blueprint = Blueprint(ROUTE_NAME, __name__)

REGISTER_LOCK = Semaphore()


class ClientTally:
    '''
    Client Count manager class
    '''

    def __init__(self):
        '''
        constructor
        '''
        self._count = 0
        self.clients_info = []

    def register_client(self, client_info, ip_address):
        '''
        register function, increments the counter and returns id
        '''
        self._count += 1

        client = {
            'name': f'client-{self._count}',
            'hostname': client_info['sysinfo']['hostname'],
            'ip_address': ip_address
        }

        self.clients_info.append(client)

        return client['name']


CLIENT_TALLY = ClientTally()


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

    data = {'id': CLIENT_TALLY.register_client(request.get_json(), request.remote_addr)}

    REGISTER_LOCK.release()
    return jsonify(data)

'''
Client Management Routing Module
'''

from flask import Blueprint, jsonify

ROUTE_NAME = 'client-manager'
blueprint = Blueprint(ROUTE_NAME, __name__)


@blueprint.route('/')
def root():
    '''
    blueprint root route, provides a brief description of the route,
    with some additional information.
    '''
    data = {'message': f'This is the \'{ROUTE_NAME}\' router.'}
    return jsonify(data)


@blueprint.route('/register')
def register():
    '''
    register route, for clients to register on the server and obtain IDs.
    '''
    data = {'message': 'This is the \'register\' router.'}
    return jsonify(data)

'''
Key Value Store Management Router
'''

from flask import Flask, jsonify, request
from Semaphore import Semaphore
from key_val_store import KeyValueStore

app = Flask(__name__)

WRITE_LOCK = Semaphore()
keyValueStore = KeyValueStore()


@app.route('/')
def root():
    '''
    app root route, provides a brief description of the route,
    with some additional information.
    '''
    data = {'message': 'This is the Key Value Store Microservice.'}
    return jsonify(data)


@app.route('/get', methods=['GET'])
def get_val():
    '''
    get value of key
    '''
    key = request.args['key']

    WRITE_LOCK.wait()

    if not keyValueStore.check(key):
        return jsonify({'res': 404})

    value = keyValueStore.get(key)

    return jsonify({'value': value, 'res': 200})


@app.route('/delete', methods=['GET'])
def delete_val():
    '''
    get value of key
    '''
    key = request.args['key']

    WRITE_LOCK.wait()

    if not keyValueStore.check(key):
        return jsonify({'value': False, 'res': 404})

    keyValueStore.delete(key)

    return jsonify({'value': True, 'res': 200})


@app.route('/set', methods=['POST'])
def set_val():
    '''
    register route, for clients to register on the server and obtain IDs.
    '''
    data = request.get_json()

    WRITE_LOCK.acquire()

    keyValueStore.set(data['key'], data['value'])

    WRITE_LOCK.release()
    return jsonify({'res': 200})


if __name__ == '__main__':
    app.run(port=6666, debug=False, host='0.0.0.0')

'''
Main entry point for the server application.
This hosts all the server routes and invokes the main function.
'''
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


if __name__ == '__main__':
    print(app.url_map)
    app.run(debug=False)

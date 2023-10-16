'''
Module hosting Application Prompts
'''
from global_kvset import app_globals


def management_server_prompt(client_state: dict):
    '''
    Prompt to ask for Management Server URL
    '''
    override_url = input(
        'Enter Management Server URL or PORT (Leave empty to use the default from .env): ').strip()

    if len(override_url) > 0:
        if override_url.isdigit():
            client_state['server_url'] = f'http://localhost:{override_url}'
        else:
            client_state['server_url'] = f'http://{override_url}'


def http_server_prompt():
    '''
    Prompt to ask for HTTP Server URL for Middleware Downstream Services
    '''

    SERVER_PORT = int(input('Enter HTTP server port: ').strip())
    app_globals.set('LOOPBACK_URL', f'http://localhost:{SERVER_PORT}')
    app_globals.set('HTTP_SERVER_PORT', SERVER_PORT)

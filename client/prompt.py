'''
Module hosting Application Prompts
'''


def management_server_prompt(client_state: dict):
    '''
    Prompt to ask for Management Server URL
    '''
    override_url = input(
        'Enter Management Server URL or PORT (Leave empty to use the default from .env): ').strip()

    override_server_url(override_url, client_state)


def override_server_url(override_url: str, client_state: dict):
    if len(override_url) > 0:
        if override_url.isdigit():
            client_state['server_url'] = f'http://localhost:{override_url}'
        else:
            client_state['server_url'] = f'http://{override_url}'
    else:
        print('Invalid Server URL/PORT')
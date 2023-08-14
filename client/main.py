import os
from helpers.client_registration import register_client
from dotenv import load_dotenv


# import environment variables
load_dotenv()

# fetch mme server url 
MANAGEMENT_SERVER = os.getenv('MANAGEMENT_SERVER')

# register client machine on mme server
CLIENT_REGINFO = register_client(MANAGEMENT_SERVER)
print(CLIENT_REGINFO)

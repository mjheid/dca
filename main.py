from bottle import Bottle

from featurecloud.api.http_ctrl import api_server
from featurecloud.api.http_web import web_server

import federated_dca.app

from featurecloud.engine.app import app

server = Bottle()
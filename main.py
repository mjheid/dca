from FeatureCloud.app.engine.app import app
from FeatureCloud.app.api.http_ctrl import api_server
from FeatureCloud.app.api.http_web import web_server
import federated_dca.app
from bottle import Bottle

if __name__ == '__main__':
    app.register()
    server = Bottle()
    server.mount('/api', api_server)
    server.mount('/web', web_server)
    server.run(host='localhost', port=5000)
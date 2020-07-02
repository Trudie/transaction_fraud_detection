from flask import Flask, json, request
from server.test_fraud import prediction

api = Flask(__name__)


@api.route('/fraud', methods=['GET'])
def get_fraud():
    data = request.get_json()

    pred_y = prediction(data)

    return json.dumps({'is_fraud': pred_y[0]})


if __name__ == '__main__':
    api.run()

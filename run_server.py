from flask import Flask
from flask_restful import Api, Resource, reqparse
from trained_model import TrainedModel as tm

app = Flask(__name__)
api = Api(app)

class GetResult(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('sentence1', type=str)
        parser.add_argument('sentence2', type=str)
        req_data = parser.parse_args()
        sent1, sent2 = req_data["sentence1"], req_data["sentence2"]
        res = tm.predict(sent1, sent2)
        return {"result": res}

# Add url resource
api.add_resource(GetResult, '/')

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0')

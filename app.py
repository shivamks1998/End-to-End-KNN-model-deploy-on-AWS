from flask import Flask,request
import pickle 
import numpy as np
import sklearn
# from sklearn.metrics import accuracy_score
app = Flask(__name__)

def model():
    global load_model 
    load_path = open("model.pkl",'rb')
    load_model = pickle.load(load_path)


@app.route("/")
def hello():
    return "hello my world"

@app.route("/predict",methods=['POST','GET'])
def output(): 
    if request.method == 'POST':
        data = request.get_json(force = True)  # Get data posted as a json\\]
        data = np.array([data])  # converts shape from (4,) to (1, 4)
        prediction = load_model.predict(data)  # runs globally loaded model on the data
        return str(prediction[0])
    else:
        return "tata bye bye"
 
    
        # prediction = load_model.predict()
if __name__ =="__main__":
    model()
    app.run(debug = True,host="0.0.0.0")

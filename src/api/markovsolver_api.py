'''
Created on 16 ago 2024

@author: Pablo Lopez
'''

from ..markovsolver import MarkovChain
    
from flask import Flask, request

import json

app = Flask(__name__)
port = 5000;
    
@app.route("/")
def check_connection():
    return "OK"

@app.route("/markovsolver",methods=['GET','POST'])
def markovsolver():
    '''
    posted_data = json.load(request.files['data'])
    posted_model_representation = json.load(request.files['model_representation'])
     '''
    
    print(request.json)
    
    posted_data = request.json['data']
    posted_model_representation = request.json['model_representation']
    
    print(posted_data)
    print(posted_model_representation)
    
    mc = MarkovChain(posted_data, posted_model_representation, completeness=True, has_consequences=True)
    print(mc.get_symbolic_system())
    time = request.json['time']
    solution = mc.get_results_by_consequences(time)
    
    reliability = {}
    reliability_else = 1;
    for k in request.json['solution_of_interest']:
        reliability[k] = solution[k]
        reliability_else = reliability_else - solution[k]
    reliability["else"] = reliability_else
    print(reliability)
    
    return reliability.__str__()
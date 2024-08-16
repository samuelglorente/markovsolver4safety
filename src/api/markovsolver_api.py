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
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/markovsolver",methods=['GET','POST'])
def markovsolver():
    
    posted_data = json.load(request.files['data'])
    posted_model_representation = json.load(request.files['model_representation'])
     
    print(posted_data)
    print(posted_model_representation)
    
    mc = MarkovChain(posted_data, posted_model_representation, completeness=True, has_consequences=True)
    print(mc.get_symbolic_system())
    time = 30
    solution = mc.get_results_by_consequences(time)
    reliability = solution['Nominal Operation'] + solution['No Redundancy']
    print(reliability)
    
    return "<p>This is Markov tool!</p>"
# Charles Ebeling - An Introduction to Reliability and Maintainability
# Created example from section 6.5 (figure 6.6)

import os
import sys
import numpy

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_path not in sys.path:
    sys.path.append(pkg_path)
    
from src.markovsolver import MarkovChain

def main(time): 
    
    data = {
    "State": ["1", "2", "3"],
    "System": ["Operating", "Failed Open", "Failed Short"],
    }

    model_representation = {
        'System': {
            'Operating': {
                'Failed Open': {'value': 0.01, 'symbol':  '\\lambda_1'},
                'Failed Short': {'value': 0.05, 'symbol':  '\\lambda_2'},
                'Operating': {}
            },
            'Failed Open': {
                'Failed Open': {}
            },
            'Failed Short': {
                'Failed Short': {}
            }
        }
    }

    # Markov Chain object + solution
    mc = MarkovChain(data, model_representation, completeness=False, has_consequences=False)
    solution = mc.get_results_by_states(time)  
    reliability = solution['1']

    return reliability

def test_time_1():
    t = 100
    assert round(main(t), 3) == round(numpy.exp(-(0.01+0.05)*t),3)
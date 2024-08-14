# Charles Ebeling - An Introduction to Reliability and Maintainability
# Example 6.7

import os
import sys

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_path not in sys.path:
    sys.path.append(pkg_path)
    
from src.markovsolver import MarkovChain

def main(time):
    
    data = {
        "State": ["1", "2", "3"],
        "Machine": ["Operating", "Degraded", "Failed"]
        }
    
    model_representation = {
        'Machine': {
            'Operating': {
                'Failed': {'value': 0.01, 'color': '#FF1744', 'symbol': '\\lambda_1'},
                'Degraded': {'value': 0.05, 'color': '#FFAB00', 'symbol': '\\lambda_2'},
                'Operating': {}
            },
            'Degraded': {
                'Failed': {'value': 0.07, 'color': '#455A64', 'symbol': '\\lambda_3'},
                'Degraded': {}
            },
            'Failed': {
                'Failed': {}
            }
        }
    }

    # Markov Chain object + solution
    mc = MarkovChain(data, model_representation, completeness=False, has_consequences=False)
    solution = mc.get_results_by_states(time)  
    return solution['1']

def test_time_1(): 
    assert round(main(1), 3) == 0.942
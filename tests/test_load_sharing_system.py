# Charles Ebeling - An Introduction to Reliability and Maintainability
# Example 6.1

import os
import sys

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_path not in sys.path:
    sys.path.append(pkg_path)
    
from src.markovsolver import MarkovChain

def main(time): 
    
    data = {
    "State": ["1", "2", "3", "4"],
    "Generator 1": ["Operating", "Failed", "Overloaded", "Failed"],
    "Generator 2": ["Operating", "Overloaded", "Failed", "Failed"]
    }
 
    model_representation = {
        'Generator 1': {
            'Operating': {
                'Failed': {'value': 0.01, 'color': '#FF1744', 'symbol': '\\lambda'},
                'Overloaded': {},
                'Operating': {}
            },
            'Overloaded': {
                'Failed': {'value': 0.1, 'color': '#F57C00', 'symbol': '\\lambda^+'},
                'Overloaded': {}
            },
            'Failed': {
                'Failed': {}
            }
        },
        'Generator 2': {
            'Operating': {
                'Failed': {'value': 0.01, 'color': '#FF1744', 'symbol': '\\lambda'},
                'Overloaded': {},
                'Operating': {}
            },
            'Overloaded': {
                'Failed': {'value': 0.1, 'color': '#F57C00', 'symbol': '\\lambda^+'},
                'Overloaded': {}
            },
            'Failed': {
                'Failed': {}
            }
        }
    }

    # Markov Chain object + solution
    mc = MarkovChain(data, model_representation, completeness=False, has_consequences=False)
    solution = mc.get_results_by_states(time)  
    reliability = solution['1'] + solution['2'] + solution['3']

    return reliability

def test_time_10():
    assert round(main(10), 4) == 0.9314
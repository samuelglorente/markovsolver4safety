# Charles Ebeling - An Introduction to Reliability and Maintainability
# Example 6.6

import os
import sys

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_path not in sys.path:
    sys.path.append(pkg_path)
    
from src.markovsolver import MarkovChain

def main(time): 
    
    data = {
    "State": ["1", "2", "3", "4"],
    "Transmitter 1": ["Operating", "Failed", "Failed", "Failed"],
    "Transmitter 2": ["Standby", "Operating", "Failed", "Failed"],
    "Transmitter 3": ["Standby", "Standby", "Operating", "Failed"]
    }
 
    model_representation = {
        'Transmitter 1': {
            'Operating': {
                'Failed': {'value': 0.0035, 'symbol':  '\\lambda'},
                'Operating': {}
            },
            'Standby': {
                'Operating': {},
                'Standby': {}
            },
            'Failed': {
                'Failed': {}
            }
        },
        'Transmitter 2': {
            'Operating': {
                'Failed': {'value': 0.0035, 'symbol':  '\\lambda'},
                'Operating': {}
            },
            'Standby': {
                'Operating': {},
                'Standby': {}
            },
            'Failed': {
                'Failed': {}
            }
        },
        'Transmitter 3': {
            'Operating': {
                'Failed': {'value': 0.0035, 'symbol':  '\\lambda'},
                'Operating': {}
            },
            'Standby': {
                'Operating': {},
                'Standby': {}
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

def test_time_500():
    assert round(main(500), 3) == 0.744
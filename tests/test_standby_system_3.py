# Charles Ebeling - An Introduction to Reliability and Maintainability
# Example 6.5

#! THIS EXAMPLE HAS SIMULTANEOUS FAILURES NOT IMPLEMENTED YET
import os
import sys

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pkg_path not in sys.path:
    sys.path.append(pkg_path)
    
from src.markovsolver import MarkovChain

def main(time): 
    
    data = {
    "State": ["1", "2", "3", "4"],
    "Active Generator": ["Operating", "Failed", "Operating", "Failed"],
    "Standby Generator": ["Standby", "Operating", "Failed", "Failed"],
    "Consequence": ["Nominal Operation", "No Redundancy", "No Redundancy", "No Operation"]
    }
 
    model_representation = {
        'Active Generator': {
            'Operating': {
                'Failed': {'value': 0.01, 'color': '#FF1744', 'symbol':  '\\lambda_1'},
                'Operating': {}
            },
            'Failed': {
                'Failed': {}
            }
        },
        'Standby Generator': {
            'Standby': {
                'Failed': {'value': 0.001, 'symbol': '\\lambda_2^-'},
                'Operating': {},
                'Standby': {}
            },
            'Operating': {
                'Failed': {'value': 0.1, 'color': '#F57C00', 'symbol': '\\lambda_2'},
                'Operating': {}
            },
            'Failed': {
                'Failed': {}
            }
        }
    }

    # Markov Chain object + solution
    mc = MarkovChain(data, model_representation, completeness=False, has_consequences=True)
    solution = mc.get_results_by_consequences(time)  
    reliability = solution['Nominal Operation'] + solution['No Redundancy']

    return reliability

def test_time_30():
    assert round(main(30), 4) == 0.8088
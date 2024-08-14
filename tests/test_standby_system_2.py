# Charles Ebeling - An Introduction to Reliability and Maintainability
# Example 6.3

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
                'Failed': {'value': 0.002, 'color': '#FF1744', 'symbol':  '\\lambda'},
                'Operating': {}
            },
            'Failed': {
                'Failed': {}
            }
        },
        'Standby Generator': {
            'Standby': {
                'Failed': {'value': 0.0001, 'symbol': '\\lambda^-'},
                'Operating': {},
                'Standby': {}
            },
            'Operating': {
                'Failed': {'value': 0.002, 'color': '#F57C00', 'symbol': '\\lambda'},
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

def test_time_100():
    assert round(main(100), 3) == 0.982
    
def test_time_150():
    assert round(main(150), 3) == 0.961
    
def test_time_173():
    assert round(main(173), 3) == 0.950
    
def test_time_175():
    assert round(main(175), 3) == 0.949
    
def test_time_200():
    assert round(main(200), 3) == 0.936
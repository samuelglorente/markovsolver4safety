'''
Created on 17 ago 2024

@author: Pablo Lopez
'''
import json
import requests

if __name__ == '__main__':

    url = "http://127.0.0.1:5000/"
    
    r = requests.get(url)
    print(str(r.content, 'utf-8'))

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
        },
        'joint_failures': [
            {
                'components': {
                    'Active Generator': {'from': 'Operating', 'to': 'Failed'},
                    'Standby Generator': {'from': 'Standby', 'to': 'Failed'}
                },
                'value': 0.01,
                'symbol': '\\lambda_1'
            }
        ],
        'state_modifiers': {
            'p': {'value': 0.1, 'modifies': {'from': '1', 'to': '4'}},
            '(1-p)': {'value': 0.9, 'modifies': {'from': '1', 'to': '2'}}
        }
    }
    
    time = 30;
    
    solution = ['Nominal Operation', 'No Redundancy']
        
    model = {'data': data, 
             'model_representation': model_representation,
             'time': time, 
             'solution_of_interest': solution}
    
    url = "http://127.0.0.1:5000/markovsolver"
    r = requests.post(url, json=model)
    print(str(r.content, 'utf-8'))
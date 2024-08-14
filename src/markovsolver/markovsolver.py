import os
import numpy
import pandas
import odeintw
# Packages only needed for MarkovChain.draw() function (optional)
import unicodeitplus
import pygraphviz as pgv


class MarkovChain:
    """_summary_

    _extended_summary_

    Parameters
    ----------


    Examples
    --------

    """
    
    def __init__(self, data: dict[str, list[str]] | pandas.DataFrame, 
                 model_representation: dict[str, dict[str, dict[str, dict[str, float | str]]]], 
                 completeness: bool = True, has_consequences: bool = False) -> None:
        """
        Parameters
        ----------
        data : dict[str, list[str]] | pandas.DataFrame
            _description_
        model_representation : dict[str, dict[str, dict[str, dict[str, float | str]]]]
            _description_
        completeness : bool, optional
            _description_, by default True
        has_consequences : bool, optional
            _description_, by default False

        Raises
        ------
        TypeError
            _description_
        KeyError
            _description_
        """

        self.data = data

        self.model_representation = model_representation
        self.parameters = {'0': {'value': 0.}}
        self.__extract_parameters(self.model_representation)
        
        self.completeness = completeness
        self.has_consequences = has_consequences

        if self.has_consequences:
            last_data_col_index = -1
            self.consequences = self.data.iloc[:, last_data_col_index].to_list()
        else:
            last_data_col_index = len(self.data)
        
        self.state_names = self.data.iloc[:, 0].to_list()
        self.systems_names = self.data.columns.to_list()[1:last_data_col_index]
        self.state_matrix = [list(row[self.systems_names]) for _, row in self.data.iterrows()]
        self.state_vectors = []
        
        self.__get_matrix()

    @property
    def data(self) -> pandas.DataFrame:
        """This property sets the main information to perform the Markov Chain analysis.

        The data can be entered as a dictionary (and will be converted to a DataFrame) or
        directly as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            Structured data to perform the Markov Chain analysis

        Raises
        ------
        ValueError
            _description_
        """

        return self._data
    
    @data.setter
    def data(self, data: dict[str, list[str]] | pandas.DataFrame) -> None:
        if type(data) is not pandas.core.frame.DataFrame:
            try:
                data = pandas.DataFrame(data)
            except ValueError:
                raise ValueError(
                    "Parameter data shall be a dictionary to construct the DataFrame"
                    )
            
        self._data = data

    @property
    def model_representation(self) -> dict[str, dict[str, dict[str, dict[str, float | str]]]]:
        return self._model_representation
    
    @model_representation.setter
    def model_representation(self, model_representation):
        if type(model_representation) is not dict:
            raise TypeError("The model representation shall be a dictionary.")
        
        def __check_modelrepresentation(d, loop = False, parent_key = None, middle_key = None):
            for key, value in d.items():
                if loop is False:
                    parent_key = key

                if key == 'joint_failures':
                    if not isinstance(value, list):
                        raise ValueError(f"Error: The value of 'joint_failures' should be a list. Found {type(value)} instead.")
                    for idx, joint_failure in enumerate(value):
                        if not isinstance(joint_failure, dict):
                            raise ValueError(f"Error: Each item in 'joint_failures' should be a dict. Found {type(joint_failure)} at index {idx}.")                        
                        if 'value' in joint_failure:
                            if not isinstance(joint_failure['value'], float):
                                print(f"Error: The 'value' in joint_failures[{idx}] is not a float. Attempting to convert...")
                                try:
                                    joint_failure['value'] = float(joint_failure['value'])
                                    print(f"'value' in joint_failures[{idx}] successfully converted to float.")
                                except ValueError:
                                    raise ValueError(f"Error: 'value' in joint_failures[{idx}] cannot be converted to float.")
                        else:
                            raise ValueError(f"Error: Missing 'value' key in joint_failures[{idx}].")
                        if 'symbol' not in joint_failure:
                            raise ValueError(f"Error: Missing 'symbol' key in joint_failures[{idx}].")
                elif key == 'state_modifiers':
                    if isinstance(value, dict):
                        if 'value' in value:
                            if not isinstance(value['value'],float):
                                print(f"Error: The value of 'value' in {key} is not a float. Attempting to convert...")
                                try:
                                    value['value'] = float(value['value'])
                                    print(f"{value['value']} is now a float.")
                                except:
                                    raise ValueError(f"{value['value']} cannot be converted to float.")
                elif isinstance(value, dict):
                    if 'value' in value:
                        if not isinstance(value['value'], float):
                            print(f"Error: The value of 'value' in {key} is not a float. Attempting to convert...")
                            try:
                                value['value'] = float(value['value'])
                                print(f"{value['value']} is now a float.")
                            except:
                                raise ValueError(f"{value['value']} cannot be converted to float.")
                        if 'symbol' not in value:
                            raise ValueError(f"Error: There is no 'symbol' parameter in {parent_key}:{middle_key}:{key}, and one has to be assigned.")
                    __check_modelrepresentation(value, loop = True, parent_key = parent_key, middle_key = key)

        __check_modelrepresentation(model_representation)

        joint_failures = model_representation.get('joint_failures')
        if joint_failures is None:
            model_representation['joint_failures'] = []
        
        state_modifiers = model_representation.get('state_modifiers')
        if state_modifiers is None:
            model_representation['state_modifiers'] = {}

        self._model_representation = model_representation

    @property
    def completeness(self):
        return self._completeness
    
    @completeness.setter
    def completeness(self, completeness):
        self._completeness = completeness

    @property
    def initial_conditions(self):
        return self._initial_conditions
    
    @initial_conditions.setter
    def initial_conditions(self, initial_conditions):
        if self.completeness:
            self._initial_conditions = initial_conditions[:-1]
        else:
            self._initial_conditions = initial_conditions
    
    @property
    def time(self):
        return self._time
    
    @time.setter
    def time(self, time):
        if type(time) == int or type(time) == float:
            self._time = numpy.array([0., float(time)])
        else:
            raise(ValueError)
    
    @property
    def size(self):
        return len(self.state_names)

    @classmethod
    def from_csv(cls, csv_path, transitions, parameters, completeness = True, csv_delimiter = ';'):

        csv_data = pandas.read_csv(csv_path, delimiter = csv_delimiter)
        return cls(csv_data, transitions, parameters, completeness)

    @classmethod
    def from_md(cls, md, transitions, parameters):
        ...

    def get_symbolic_system(self):
        if self.completeness:
            state_names = self.state_names[:-1]
        else:
            state_names = self.state_names
        matrix = self.__get_symbolic_matrix()

        # First Array (dPi/dt)
        latex_text = ''
        latex_text += '\\left[\\begin{array}{c}'
        for state in state_names:
            latex_text += f"dP_{{{state}}}(t)/dt"
            if state != state_names[-1]:
                latex_text += '\\\\'
        latex_text += '\\end{array}\\right]'
        
        latex_text += '='
        
        # Matrix
        latex_text += '\\begin{bmatrix}'
        for i, row in enumerate(matrix):
            for j, col in enumerate(row):
                latex_text += matrix[i][j]
                latex_text += '&'
            if latex_text.endswith('&'):
                latex_text = latex_text[:-1]
            latex_text += '\\\\'
        if latex_text.endswith('\\\\'):
            latex_text = latex_text[:-2]
        latex_text += '\\end{bmatrix}'
        
        latex_text += '\\cdot'
        
        # Second Array (Pi)
        latex_text += '\\left[\\begin{array}{c}'
        for state in state_names:
            latex_text += f"P_{{{state}}}(t)"
            if state != state_names[-1]:
                latex_text += '\\\\'
        latex_text += '\\end{array}\\right]'
        
        return latex_text
    
    def get_results_by_states(self, time, initial_conditions = None):
        results = self.__solve(time, initial_conditions)
        return results['by_states']

    def get_results_by_consequences(self, time, initial_conditions = None):
        results = self.__solve(time, initial_conditions)
        return results['by_consequences']

    def get_graph_data(self):
         
        graph_data = {
            'nodes': self.state_names,
            'edges': []
        }
        for i in range(self.size):
            for j in range(self.size):
                if self.state_vectors[i][j] != '0': 
                    graph_data['edges'].append((
                        self.state_names[i], 
                        self.state_names[j], 
                        self.state_vectors[i][j]
                    ))
        return graph_data

    def draw(self, img_path = None):
        """
        Draw the Markov Chain
        """
        graph = pgv.AGraph(directed = True, strict = False, rankdir="LR")
        
        # Draw the nodes
        for node in self.state_names:
            graph.add_node(node)
            
        # Add the transitions
        for i in range(self.size):
            for j in range(self.size):
                if self.state_vectors[i][j] != '0': 
                    prob = unicodeitplus.replace(self.state_vectors[i][j])
                    graph.add_edge(
                        self.state_names[i], 
                        self.state_names[j], 
                        color = self.parameters[self.state_vectors[i][j]].get('color', 'black'),
                        label = prob,
                        fontcolor = self.parameters[self.state_vectors[i][j]].get('color', 'black')
                        )
        
        graph.layout('dot')
        
        if img_path is None:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            img_path = f'{dir_path}\\markov.svg'
            
        graph.draw(img_path)

    def __extract_parameters(self, data):
        for key, value in data.items():
            if key == 'joint_failures':
                for joint_failure in value:
                    if 'symbol' in joint_failure and 'value' in joint_failure:
                        symbol = joint_failure['symbol']
                        self.parameters[symbol] = {k: v for k, v in joint_failure.items() if k != 'symbol' and k != 'components'}
            elif isinstance(value, dict):
                if 'symbol' in value and 'value' in value:
                    symbol = value['symbol']
                    self.parameters[symbol] = {k: v for k, v in value.items() if k != 'symbol'}
                self.__extract_parameters(value)

    def __get_transitions(self):

        for i in range(self.size):
            local_state_transition = ['0'] * self.size
            for j in range(self.size):
                check_transition = self.__check_transition([i, j])
                if check_transition[0]:
                    if check_transition[1] == 'single':
                        system_transiting = list(check_transition[2][0].keys())[0]
                        from_state_st = check_transition[2][0][system_transiting]['from_state']
                        to_state_st = check_transition[2][0][system_transiting]['to_state']
                        local_state_transition[j] = self.model_representation[system_transiting][from_state_st][to_state_st]['symbol']
                    elif check_transition[1] == 'multiple':
                        system_transiting = 'joint_failures'
                        joint_transition_index = check_transition[2]
                        local_state_transition[j] = self.model_representation[system_transiting][joint_transition_index]['symbol']
            
            self.state_vectors.append(local_state_transition) 
        
        # Modifiers
        modifiers = self.model_representation.get('state_modifiers')
        if modifiers != {}:
            for modifier, modifier_content in modifiers.items():
                row_mod = modifier_content['modifies']['from']
                col_mod = modifier_content['modifies']['to']
                mod_value = modifier_content['value']

                row_mod_idx = self.state_names.index(row_mod)
                col_mod_idx = self.state_names.index(col_mod)

                old_parameter = self.state_vectors[row_mod_idx][col_mod_idx]
                new_parameter = modifier + old_parameter
                self.state_vectors[row_mod_idx][col_mod_idx] = new_parameter
                self.parameters[new_parameter] = {'value': self.parameters[old_parameter]['value']*mod_value}
                if self.parameters[old_parameter].get('color') is not None:
                    self.parameters[new_parameter]['color'] = self.parameters[old_parameter]['color']
        
    def __check_transition(self, indexes):
        n_systems = len(self.systems_names)
        i = indexes[0]
        j = indexes[1]

        transition_possible = True
        transition_accomplished = []
        transition_system_info = []
        transition_result = []

        for k in range(n_systems):
            from_state = self.model_representation[self.systems_names[k]].get(self.state_matrix[i][k])
            if from_state is not None:
                to_state = from_state.get(self.state_matrix[j][k])
                if to_state is not None:
                    if to_state != {}:
                        transition_accomplished.append(True)
                        transition_system_info.append({self.systems_names[k]: {'from_state': self.state_matrix[i][k], 'to_state': self.state_matrix[j][k]}})
                    else:
                        transition_accomplished.append(False)
                        transition_system_info.append(None)
                else:
                    transition_accomplished.append(False)
                    transition_possible = False
                    transition_system_info.append(None)
        
        if any(transition_accomplished) and transition_possible:
            true_systems = [transition_system_info[index] for index, value in enumerate(transition_accomplished) if value]
            if len(true_systems) == 1:
                transition_result = [True, 'single', true_systems]
            else:
                joint_failures = self.model_representation.get('joint_failures')
                if joint_failures != []:
                    multiple_transition_bool, multiple_transition_index = self.__check_transitions_in_joint_failures(true_systems, joint_failures)
                    if multiple_transition_bool:
                        transition_result = [True, 'multiple', multiple_transition_index]
                    else:
                        transition_result = [False, '', []]
                else:
                    transition_result = [False, '', []]
        else:
            transition_result = [False, '', []]

        return transition_result

    def __check_transitions_in_joint_failures(self, transitions, joint_failures):
        for index, failure in enumerate(joint_failures):
            if all(failure['components'].get(generator) == {'from': states['from_state'], 'to': states['to_state']}
                for transition in transitions
                for generator, states in transition.items()):
                return True, index
        return False, None

    def __get_matrix(self):

        self.__get_transitions()
        
        matrix = numpy.zeros((self.size, self.size), dtype=float)
        
        index = 0
        for array in self.state_vectors:
            for col in range(self.size):
                matrix[index][index] -= self.parameters[array[col]]['value']
                if col < self.size:
                    matrix[col][index] += self.parameters[array[col]]['value']
            index += 1
                
        if self.completeness:
            matrix = numpy.delete(matrix, self.size - 1, axis=0)
            matrix = numpy.delete(matrix, self.size - 1, axis=1)
            
        self.matrix = matrix
        
    def __get_symbolic_matrix(self):
        matrix = numpy.empty((self.size, self.size), dtype=numpy.dtype('U100')) # type: ignore
        
        for i in range(len(self.state_vectors)):
            diagonal = ''
            count = 0
            for j in range(self.size):
                if self.state_vectors[i][j] != '0':
                    count += 1
                    diagonal +=  self.state_vectors[i][j] + '+'
                    if j < self.size:
                        matrix[j][i] += self.state_vectors[i][j]
            if diagonal.endswith('+'):
                diagonal = diagonal[:-1]
            if diagonal != '':
                if count > 1:
                    diagonal = f'-({diagonal})'
                else:
                    diagonal = f'-{diagonal}'
            matrix[i][i] = diagonal 
        matrix[matrix==''] = '0'

        if self.completeness:
            matrix = numpy.delete(matrix, self.size - 1, axis=0)
            matrix = numpy.delete(matrix, self.size - 1, axis=1)
        
        return matrix        
    
    @staticmethod
    def __psys(p, t, m):
        return m.dot(p)
    
    def __solve(self, time, initial_conditions = None):
        
        self.time = time
        if initial_conditions is None:
            if self.completeness:
                initial_conditions = numpy.array([1] + [0] * (self.size - 2))
            else:
                initial_conditions = numpy.array([1] + [0] * (self.size - 1))

        solution = odeintw.odeintw(self.__psys, initial_conditions, self.time, args=(self.matrix,))[-1]

        if self.completeness:
            solution = numpy.append(solution, 1. - numpy.sum(solution))

        solution_dict = {
            'by_states': {self.state_names[i]: float(solution[i]) for i in range(self.size)},
            'by_consequences': {}
        }

        if self.has_consequences:
            for index, consequence in enumerate(self.consequences):
                solution_dict['by_consequences'].setdefault(consequence, 0)
                solution_dict['by_consequences'][consequence] += float(solution[index])
        
        return solution_dict


if __name__ == '__main__':
    
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
    
    mc = MarkovChain(data, model_representation, completeness=True, has_consequences=True)
    print(mc.get_symbolic_system())
    time = 30
    solution = mc.get_results_by_consequences(time)
    reliability = solution['Nominal Operation'] + solution['No Redundancy']
    print(reliability)

    #mc.draw()
    #print(mc.get_graph_data())

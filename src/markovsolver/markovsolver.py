import os
import numpy
import pandas
import odeintw
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
    
    def __init__(self, data: dict[str, list[str]] | pandas.DataFrame, config: dict[str, dict[str,list[str]]], 
                 completeness: bool = True, has_consequences: bool = False) -> None:
        """
        Parameters
        ----------
        data : dict[str, list[str]] | pandas.DataFrame
            _description_
        config : dict[str, dict[str,list[str]]]
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
        
        if type(config) is not dict:
            raise TypeError("The config parameter shall be a dictionary.")
        elif not config.get('parameters'):
            raise KeyError("The config parameter is set incorrectly. Parameters key is missing.")
        elif not config.get('transitions'):
            raise KeyError("The config parameter is set incorrectly. Transitions key is missing.")
        else:
            self.parameters = config['parameters']
            self.transitions = config['transitions']

        
        self.completeness = completeness
        self.has_consequences = has_consequences

        if self.has_consequences:
            last_data_col_index = -2
            self.consequences = self.data.iloc[:, last_data_col_index].to_list()
        else:
            last_data_col_index = -1
        
        
        self.state_names = self.data.iloc[:, 0].to_list()
        self.systems_names = self.data.columns.to_list()[1:last_data_col_index]

        self.state_transitions = []
        for state_transition in self.data.iloc[:,-1].to_list():
            self.state_transitions.append(''.join([str(x) for x in state_transition.split(",")]).split())

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
    def parameters(self) -> dict[str, dict[str, int | str]]:
        """_summary_

        _extended_summary_

        Returns
        -------
        dict[str, dict[str, int | str]]
            _description_

        Raises
        ------
        TypeError
            _description_
        KeyError
            _description_
        ValueError
            _description_
        """

        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters):
        if type(parameters) is not dict:
            raise TypeError("The parameters element shall be a dictionary.")
        else:

            for key in parameters:
                if not parameters[key].get('value'):
                    raise KeyError (f"Parameter {key} does not have a designated value.")
                else:
                    value = parameters[key].get('value')
                    if type(value) != float:
                        raise ValueError (f"Parameter {key} does not have a number format type.")
            
            parameters['0'] = {'value': 0.}

        self._parameters = parameters

    @property
    def transitions(self) -> dict[str, dict[str, dict[str, str]]]:
        return self._transitions
    
    @transitions.setter
    def transitions(self, transitions):
        if type(transitions) is not dict:
            raise TypeError("The transitions element shall be a dictionary.")
        
        self._transitions = transitions

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
                        self.__latextounicode(self.state_vectors[i][j]) #? LaTeXor unicode?
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
                    prob = self.__latextounicode(self.state_vectors[i][j])
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

    def __get_matrix(self):

        self.state_matrix = [list(row[self.systems_names]) for _, row in self.data.iterrows()]

        self.state_vectors = []
        for index, state_transition in enumerate(self.state_transitions):
            local_state = ['0'] * self.size
            for transition in state_transition:
                transition_index = self.state_names.index(transition)
                col = 0
                for from_state, to_state in zip(self.state_matrix[index], self.state_matrix[transition_index]):
                    if from_state != to_state:
                        if self.transitions[self.systems_names[col]][from_state]['To'] == to_state:
                            local_state[transition_index] = self.transitions[self.systems_names[col]][from_state]['How']
                    col += 1
            self.state_vectors.append(local_state)
        
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
            initial_conditions = numpy.array([1] + [0] * (self.size - 1))
        self.initial_conditions = initial_conditions

        solution = odeintw.odeintw(self.__psys, self.initial_conditions, self.time, args=(self.matrix,))[-1]

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
    
    @staticmethod
    def __latextounicode(latex_text):
        return unicodeitplus.replace(latex_text)


if __name__ == '__main__':
    
    data = {
    "State": ["1", "2", "3", "4"],
    "Active Generator": ["Operating", "Failed", "Operating", "Failed"],
    "Standby Generator": ["Standby", "Operating", "Failed", "Failed"],
    "Consequence": ["Nominal Operation", "No Redundancy", "No Redundancy", "No Operation"],
    "Transitions": ["2, 3", "4", "4", ""]
    }
    
    config = {
        'parameters': {
            '\\lambda_1': {'value': 0.01, 'color': '#FF1744'},
            '\\lambda_2': {'value': 0.1, 'color': '#F57C00'},
            '\\lambda_2^-': {'value': 0.001}
        },
        'transitions': {
            'Active Generator': {
                'Operating': {
                    'To': 'Failed',
                    'How': '\\lambda_1'
                }
            },
            'Standby Generator': {
                'Standby': {
                    'To': 'Failed',
                    'How': '\\lambda_2^-'
                },
                'Operating': {
                    'To': 'Failed',
                    'How': '\\lambda_2'
                }
            }
        }
    }
    #config = {'parameters': {'aa': {'uu':'oo'}}, 'transitions': {'aa': 'uu'}}
    mc = MarkovChain(data, config, completeness=False, has_consequences=True)

    time = 30

    solution = mc.get_results_by_states(time)
    solution2 = mc.get_results_by_consequences(time)
    print(solution)
    print(solution2)
    #mc.draw()
    print(mc.get_graph_data())
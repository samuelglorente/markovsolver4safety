import os
import numpy
import pandas
import odeintw


class MarkovChain:
    """ Represents a Markov Chain.

    The MarkovChain class is designed to model and analyze stochastic processes using the principles of 
    Markov chains. Markov chains are mathematical systems that undergo transitions from one state to another 
    in a state space, where the probability of each subsequent state depends only on the current state 
    and not on the sequence of events that preceded it.

    This class provides a framework for defining the states of the system, the transition probabilities 
    between states, and the rates at which these transitions occur. It can be used to model a wide variety 
    of real-world processes, specifically reliability and safety analysis.

    Parameters
    ----------
    data : dict[str, list[str]] | pandas.DataFrame
        Table containing the information of the different states (like combination of operational 
        scenarios of different systems), and optionally the safety consequences of each state.
    model_representation : dict[str, dict[str, dict[str, dict[str, float  |  str]]]]
        Structured information of the system(s) state(s), transition(s) between states and rates of
        occurrence of those transition(s).
    completeness : bool, optional
        Boolean that establishes if the probability of occurrence of the last state (usually the one 
        related to the complete failed one) is calculated as 1 minus the sum of the other states 
        probabilities, by default this value is True.
    has_consequences : bool, optional
        Boolean that determines if in data there is a column with the safety consequences of each
        state, by default this value is False.
    
    Attributes
    ----------
    data : dict[str, list[str]] | pandas.DataFrame
        Sets the main information to perform the Markov Chain analysis.
    model_representation : dict[str, dict[str, dict[str, dict[str, float  |  str]]]]
        Structured information of the system(s) state(s), transition(s) between states and rates of
        occurrence of those transition(s).
    completeness : bool
        Boolean that establishes if the probability of occurrence of the last state (usually the one 
        related to the complete failed one) is calculated as 1 minus the sum of the other states 
        probabilities, by default this value is True.
    has_consequences : bool
        Boolean that determines if in data there is a column with the safety consequences of each
        state, by default this value is False.
    parameters : dict[str, dict[str, float | str]]
        Structured data that contains the summary of the transition parameters gathered from the 
        model representation.
    state_names : list[str]
        List of the names given to each state and represents a node in the Markov Graph.
    systems_names : list[str]
        List of the names of the elements (systems, components, ...) of the model.
    state_matrix : list[list[str]]
        Matrix that represents the status of each state, extracted from the attribute data.
    state_vectors : list[list[str]]
        Set of vectors that represents the transitions of each state to others.
    matrix : numpy.ndarray
        Matrix that represents the differencial equation system.
    size : int
        Number of states
    
    Methods 
    -------
    from_csv(csv_path, model_representation, completeness = True, has_consequences = False, csv_delimiter = ';')
        Initializes a MarkovChain object obtaining the variable 'data' from a csv file.
    from_md(md, model_representation, completeness = True, has_consequences = False)
        Initializes a MarkovChain object obtaining the variable 'data' from a MarkDown string.
    get_symbolic_system()
        Gets the mathematical expression of the differential equation system in LaTeX
    get_results_by_states(time, initial_conditions = None)
        Gets the probability of being in each of the states by a given time under specific initial conditions.
    get_results_by_consequences(time, initial_conditions = None)
        Gets the probability of resulting in each of the established consequences by a given time under specific 
        initial conditions.
    get_graph_data()
        Gets the nodes (states) and the edges (transitions) of the Markov Chain.
    draw(img_path = None)
        Draws the Markov Chain.

    Examples
    --------
    ** Basic Initialization **

    >>> data = {
    ...     "State": ["1", "2", "3", "4"],
    ...     "Active Generator": ["Operating", "Failed", "Operating", "Failed"],
    ...     "Standby Generator": ["Standby", "Operating", "Failed", "Failed"],
    ...     "Consequence": ["Nominal Operation", "No Redundancy", "No Redundancy", "No Operation"]
    ... }
    >>> model_representation = {
    ...     'Active Generator': {
    ...         'Operating': {
    ...             'Failed': {'value': 0.01, 'color': '#FF1744', 'symbol':  '\\lambda_1'},
    ...             'Operating': {}
    ...         },
    ...         'Failed': {
    ...             'Failed': {}
    ...         }
    ...     },
    ...     'Standby Generator': {
    ...         'Standby': {
    ...             'Failed': {'value': 0.001, 'symbol': '\\lambda_2^-'},
    ...             'Operating': {},
    ...             'Standby': {}
    ...         },
    ...         'Operating': {
    ...             'Failed': {'value': 0.1, 'color': '#F57C00', 'symbol': '\\lambda_2'},
    ...             'Operating': {}
    ...         },
    ...         'Failed': {
    ...             'Failed': {}
    ...         }
    ...     },
    ...     'joint_failures': [
    ...         {
    ...             'components': {
    ...                 'Active Generator': {'from': 'Operating', 'to': 'Failed'},
    ...                 'Standby Generator': {'from': 'Standby', 'to': 'Failed'}
    ...             },
    ...             'value': 0.01,
    ...             'symbol': '\\lambda_1'
    ...         }
    ...     ],
    ...     'state_modifiers': {
    ...         'p': {'value': 0.1, 'modifies': {'from': '1', 'to': '4'}},
    ...         '(1-p)': {'value': 0.9, 'modifies': {'from': '1', 'to': '2'}}
    ...     }
    ... }
    >>> mc = MarkovChain(data, model_representation, has_consequences=True)
    >>> print(mc.size)
    4

    ** Initialize from a csv file **
    >>> data_csv_path = 'path/to/data.csv' # Assuming it is equivalent to 'data' above
    >>> model_representation = { ... } # Assuming it is equal to the value above
    >>> mc = MarkovChain.from_csv(data_csv_path, model_representation)
    >>> print(mc.size)
    4

    ** Get the equation system in LaTeX **
    >>> print(mc.get_symbolic_system())
    \left[\begin{array}{c}dP_{1}(t)/dt\\dP_{2}(t)/dt\\dP_{3}(t)/dt\end{array}\right]=\begin{bmatrix}-((1-p)\lambda_1+\lambda_2^-+p\lambda_1)&0&0\\(1-p)\lambda_1&-\lambda_2&0\\\lambda_2^-&0&-\lambda_1\end{bmatrix}\cdot\left[\begin{array}{c}P_{1}(t)\\P_{2}(t)\\P_{3}(t)\end{array}\right]\newline P_{4}(t) = 1 - \sum_{i=1}^{3}P_{i}(t)

    ** Get graph data **
    >>> print(mc.get_graph_data)
    {'nodes': ['1', '2', '3', '4'], 'edges': [('1', '2', '(1-p)\\lambda_1'), ('1', '3', '\\lambda_2^-'), ('1', '4', 'p\\lambda_1'), ('2', '4', '\\lambda_2'), ('3', '4', '\\lambda_1')]}

    ** Draw the Markov Chain in the directory of execution (as markov.svg) **
    >>> mc.draw()

    ** Draw the Markov Chain in a specific path in other format than svg **
    >>> mc.draw(img_path = 'path/to/markov_chain.png')

    ** Solve Markov Chain and get the info for each state ** 
    >>> print(mc.get_results_by_states(30))
    {'1': 0.7189237345889834, '2': 0.06766551029955292, '3': 0.021894487072743454, '4': 0.19151626803872024}

    ** Solve Markov Chain and get the info for each consequence **
    >>> print(mc.get_results_by_consequences(30))
    {'Nominal Operation': 0.7189237345889834, 'No Redundancy': 0.08955999737229638, 'No Operation': 0.19151626803872024}

    Raises
    ------
    ImportError
        If `draw` function is used without the required optional dependencies
    """
    
    def __init__(self, data: dict[str, list[str]] | pandas.DataFrame, 
                 model_representation: dict[str, dict[str, dict[str, dict[str, float | str]]]], 
                 completeness: bool = True, has_consequences: bool = False) -> None:

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
        self.matrix = numpy.zeros((self.size, self.size), dtype=float)
        
        self.__get_matrix()

    @property
    def data(self) -> pandas.DataFrame:
        """Sets the main information to perform the Markov Chain analysis.

        It represents a table containing the information of the different states (like combination 
        of operational scenarios of different systems), and optionally the safety consequences of 
        each state. The data can be entered as a dictionary (and will be converted to a DataFrame) or
        directly as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            Structured data to perform the Markov Chain analysis

        Raises
        ------
        ValueError
            If the input value is not a pandas.DataFrame but neither a dictionary to convert
            it to a pandas.DataFrame
        """

        return self._data
    
    @data.setter
    def data(self, data: dict[str, list[str]] | pandas.DataFrame) -> None:
        if type(data) is not pandas.core.frame.DataFrame:
            try:
                data = pandas.DataFrame(data)
            except ValueError:
                raise ValueError(
                    "Parameter 'data' shall be a dictionary to construct the DataFrame"
                    )
            
        self._data = data

    @property
    def model_representation(self) -> dict[str, dict[str, dict[str, dict[str, float | str]]]]:
        """Structured information of the system(s) state(s), transition(s) between states and rates of
        occurrence of those transition(s).

        It contains two reserved keys named 'joint_failures' and 'state_modifiers' (optional). The first one 
        is used to represent the transitions that have simultaneous failures of different elements of the model. 
        The second reserved key is to 'manually' modify a transition (e.g., adding a specific factor only
        for a transition that you do not want to be general).
        
        Returns
        -------
        dict[str, dict[str, dict[str, dict[str, float | str]]]]
            System model representation

        Raises
        ------
        TypeError
            If the input is not a dictionary
        ValueError
            - If the reserved key 'joint_failures' has not a list as value.
            - If each item of the list in the reserved key 'joint_failures' is not a dictionary.
            - If the key 'value' inside a dictionary item of the list in the reserved key 'joint_failures' is not a float and cannot be converted to it.
            - If there is no 'value' key inside a dictionary item of the list in the reserved key 'joint_failures'.
            - If there is no 'symbol' key inside a dictionary item of the list in the reserved key 'joint_failures'.
            - If the values of the keys inside the reserved key 'state_modifiers' have not a float as 'value' key and cannot be converted to it.
            - If the values of the keys inside the systems representation have not a float as 'value' key and cannot be converted to it.
            - If the values of the keys inside the systems representation have a 'value' key but not a 'symbol' key.
        """
        return self._model_representation
    
    @model_representation.setter
    def model_representation(self, model_representation: dict[str, dict[str, dict[str, dict[str, float | str]]]]) -> None:
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
    def completeness(self) -> bool:
        """
        Boolean that establishes if the probability of occurrence of the last state (usually the one 
        related to the complete failed one) is calculated as 1 minus the sum of the other states 
        probabilities, by default this value is True.

        Returns
        -------
        bool
            Confirmation on how to calculate the last state probabilities
        
        Raises
        ------
        TypeError
            If the input is not a boolean
        """
        return self._completeness
    
    @completeness.setter
    def completeness(self, completeness: bool) -> None:
        if type(completeness) is not bool:
            raise TypeError("The completeness parameter shall be a boolean.")
        self._completeness = completeness
    
    @property
    def size(self) -> int:
        """Number of states

        Returns
        -------
        int
            Number of states
        """
        return len(self.state_names)

    @classmethod
    def from_csv(cls, csv_path: str, model_representation: dict[str, dict[str, dict[str, dict[str, float | str]]]], 
                 completeness: bool = True, has_consequences: bool = False, csv_delimiter: str = ';') -> 'MarkovChain':
        """Initializes a MarkovChain object obtaining the variable 'data' from a csv file.

        The MarkovChain object represents a Markov Chain used for probabilistic calculations in RAMS analysis.

        Parameters
        ----------
        csv_path : str
            Path to the csv file where the data of the table where the states, systems, statuses are storage.
        model_representation : dict[str, dict[str, dict[str, dict[str, float  |  str]]]]
            Structured information of the system(s) state(s), transition(s) between states and rates of
            occurrence of those transition(s).
        completeness : bool, optional
            Boolean that establishes if the probability of occurrence of the last state (usually the one 
            related to the complete failed one) is calculated as 1 minus the sum of the other states 
            probabilities, by default this value is True.
        has_consequences : bool, optional
            Boolean that determines if in data there is a column with the safety consequences of each
            state, by default this value is False.
        csv_delimiter : str, optional
            Delimiter of the csv file, by default this value is equal to ';'.

        Returns
        -------
        MarkovChain
            A new instance of 'MarkovChain' initialized from a csv

        Examples
        --------
        >>> data_csv_path = 'path/to/data.csv'
        >>> model_representation = {
        ...     'system1': {
        ...         'state1': {
        ...             'state2': {'value': 0.1, 'symbol': '\\lambda_1'}
        ...         },
        ...         'state2': {
        ...             'state3': {'value': 0.2, 'symbol': '\\lambda_2'}
        ...         }
        ...     }
        ... }
        >>> markov_chain = MarkovChain.from_csv(data_csv_path, model_representation)
        >>> print(markov_chain)
        <MarkovChain object at 0x...>
        """
        csv_data = pandas.read_csv(csv_path, delimiter = csv_delimiter)
        return cls(csv_data, model_representation, completeness = completeness, has_consequences = has_consequences)

    @classmethod
    def from_md(cls, md: str, model_representation: dict[str, dict[str, dict[str, dict[str, float | str]]]], 
                completeness: bool = True, has_consequences: bool = False) -> 'MarkovChain':
        """Initializes a MarkovChain object obtaining the variable 'data' from a MarkDown string.

        The MarkovChain object represents a Markov Chain used for probabilistic calculations in RAMS analysis.

        Parameters
        ----------
        md : str
            MarkDown string that represents the data of the table where the states, systems, statuses are storage.
        model_representation : dict[str, dict[str, dict[str, dict[str, float  |  str]]]]
            Structured information of the system(s) state(s), transition(s) between states and rates of
            occurrence of those transition(s).
        completeness : bool, optional
            Boolean that establishes if the probability of occurrence of the last state (usually the one 
            related to the complete failed one) is calculated as 1 minus the sum of the other states 
            probabilities, by default this value is True.
        has_consequences : bool, optional
            Boolean that determines if in data there is a column with the safety consequences of each
            state, by default this value is False.

        Returns
        -------
        MarkovChain
            A new instance of 'MarkovChain' initialized from a MarkDown string
        
        Examples
        --------
        #! TBD
        """
        md_data = None
        ...
        print("This function has not been developed yet")
        #return cls(md_data, model_representation, completeness = completeness, has_consequences = has_consequences)

    def get_symbolic_system(self) -> str:
        """Gets the mathematical expression of the differential equation system in LaTeX.

        Returns
        -------
        str
            LaTeX string that represents the differential equation system
        """

        if self.completeness:
            state_names = self.state_names[:-1]
            last_formula = '\\newline '
            last_formula += f'P_{{{self.state_names[-1]}}}(t) = 1 - \\sum_{{i=1}}^{{{self.size-1}}}P_{{i}}(t)'
        else:
            state_names = self.state_names
            last_formula = ''
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
            for j, _ in enumerate(row):
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

        latex_text += last_formula
        
        return latex_text
    
    def get_results_by_states(self, time: float, initial_conditions: numpy.ndarray = None) -> dict[str, float]:
        """Gets the probability of being in each of the states by a given time under specific initial conditions.

        Parameters
        ----------
        time : float
            Time at which the Markov Chain is analyzed
        initial_conditions : numpy.ndarray, optional
            Probability conditions where each state is at time 0, by default the value is set as None, which
            sets the first state as start point (i.e., probability equal to 1)

        Returns
        -------
        dict[str, float]
            Each of the states with their corresponding probability value
        """
        results = self.__solve(time, initial_conditions)
        return results['by_states']

    def get_results_by_consequences(self, time: float, initial_conditions: numpy.ndarray = None) -> dict[str, float]:
        """Gets the probability of resulting in each of the established consequences by a given time under specific 
        initial conditions.

        When the consequences are described, each of the states have one associated to them (several states could have
        the same consequences). Hence, this function returns the probabilities grouped by consequences.

        Parameters
        ----------
        time : float
            Time at which the Markov Chain is analyzed
        initial_conditions : numpy.ndarray, optional
            Probability conditions where each state is at time 0, by default the value is set as None, which
            sets the first state as start point (i.e., probability equal to 1)

        Returns
        -------
        dict[str, float]
            Each of the consequences with their corresponding probability value
        """
        results = self.__solve(time, initial_conditions)
        return results['by_consequences']

    def get_graph_data(self) -> dict[str, list[str | tuple[str]]]:
        """Gets the nodes (states) and the edges (transitions) of the Markov Chain.

        The transitions are represented by a tuple with three values (out state, in state, 
        transition symbol - usually the failure or repair rate).

        Returns
        -------
        dict[str, list[str | tuple[str]]]
            A list of the nodes and a list of the edges (transitions), with the out node, the in node and the 
            'symbol' value of the transition as per model representation
        """
         
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

    def draw(self, img_path: str = None) -> None:
        """Draws the Markov Chain.

        Draws a graphical representation of the Markov Chain using Graphviz and saves it in a directory.

        Parameters
        ----------
        img_path : str, optional
            Path where the graphical representation of the Markov Chain is saved, by default the value is 
            set as None which will save the image as 'markov.svg' in the execution directory.

        Raises
        ------
        ImportError
            If this function is used without the required optional dependencies
        """

        try:
            import unicodeitplus
            import pygraphviz as pgv
        except ImportError:
            raise ImportError(
                "To use the draw feature you need to install the optional dependencies 'pygraphviz' and 'unicodeitplus."
                "You can do that by executing: pip install markovsolver4safety[drawing_feature]"
            )
         
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

    def __extract_parameters(self, model_representation: dict[str, dict[str, dict[str, dict[str, float | str]]]]) -> None:
        """Extracts the transition parameters info from the model variable.

        Structures the parameter data contained in the model representation variable in a simpler way for internal
        calculations.

        Parameters
        ----------
        model_representation : dict[str, dict[str, dict[str, dict[str, float  |  str]]]]
            Structured information of the system(s) state(s), transition(s) between states and rates of
            occurrence of those transition(s).
        """

        for key, value in model_representation.items():
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

    def __get_transitions(self) -> None:
        """Gets the state_vectors attribute from the input data.
        """

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
        
    def __check_transition(self, indexes: list[str]) -> list[bool | str | list[dict[str, dict[str, str]]]]:
        """Checks if a transition is possible.

        Parameters
        ----------
        indexes : list[str]
            Indexes of the in state and out state.

        Returns
        -------
        list[bool | str | dict[str, dict[str, str]]]
            List with three elements: a boolean indicating if the transition is possible, and if possible a string
            indicating if there are only one failure or simultaneous failures and the systems/states info.
        """
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

    def __check_transitions_in_joint_failures(self, transitions: list[dict[str, dict[str, str]]], joint_failures: list[dict]) -> tuple[bool, int | None]:
        """Checks if the given transitions match any of the joint failure scenarios.

        This method evaluates whether a set of transitions corresponds to one of the pre-defined joint failures 
        within the system. Joint failures involve multiple components failing simultaneously, and each such 
        scenario is represented by a set of state transitions.

        Parameters
        ----------
        transitions : list[dict[str, dict[str, str]]]
            A list of dictionaries representing the transitions for each system involved. Each dictionary contains 
            the system name as the key, with a nested dictionary showing the 'from' and 'to' states.
        joint_failures : list[dict]
            A list of joint failure scenarios. Each scenario is represented as a dictionary that specifies the 
            components involved and their respective state transitions.

        Returns
        -------
        tuple[bool, int | None]
            A tuple where the first element is a boolean indicating whether the transitions match a joint failure 
            scenario, and the second element is the index of the matching joint failure scenario in the 
            `joint_failures` list. If no match is found, the second element is `None`.
        """

        for index, failure in enumerate(joint_failures):
            if all(failure['components'].get(generator) == {'from': states['from_state'], 'to': states['to_state']}
                for transition in transitions
                for generator, states in transition.items()):
                return True, index
        return False, None

    def __get_matrix(self) -> None:
        """Gets the transition matrix of the Markov Chain, representing the rates of transitions between states.
        """

        self.__get_transitions()
                
        index = 0
        for array in self.state_vectors:
            for col in range(self.size):
                self.matrix[index][index] -= self.parameters[array[col]]['value']
                if col < self.size:
                    self.matrix[col][index] += self.parameters[array[col]]['value']
            index += 1
                
        if self.completeness:
            self.matrix = numpy.delete(self.matrix, self.size - 1, axis=0)
            self.matrix = numpy.delete(self.matrix, self.size - 1, axis=1)
                    
    def __get_symbolic_matrix(self) -> str:
        """Gets the mathematical expresion in LaTeX of the transition matrix of the Markov Chain, 
        representing the rates of transitions between states.

        Returns
        -------
        str
            LaTeX expresion of the transition matrix of the Markov Chain
        """
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
    def __psys(p: numpy.ndarray, t: float, m: numpy.ndarray) -> numpy.ndarray:
        """Computes the product of the transition matrix and the probability vector.

        This function is used in the numerical integration process to solve the system 
        of differential equations representing the Markov Chain.

        Parameters
        ----------
        p : numpy.ndarray
            The probability vector representing the probabilities of each state at a given time
        t : float
            The current time in the integration process. This parameter is included to match the signature 
            expected by the ODE solver but is not used in the computation.
        m : numpy.ndarray
            The transition matrix of the Markov Chain, representing the rates of transitions between states.

        Returns
        -------
        numpy.ndarray
            The result of the matrix-vector multiplication `m.dot(p)`, which gives the rate of change of the 
            probability vector `p`.
        """
        return m.dot(p)
    
    def __solve(self, time: float, initial_conditions: numpy.ndarray = None) -> dict[dict[str, float]]:
        """Solves the system of differential equations for the Markov Chain over the specified time.

        This method computes the probabilities of being in each state at a specific time `t`, given initial 
        conditions and the transition matrix of the Markov Chain.

        Parameters
        ----------
        time : float
            The time at which the solution is computed. It is converted to an array `[0., time]` representing 
            the time span for the ODE solver.
        initial_conditions : numpy.ndarray, optional
            The initial state vector representing the starting probabilities for each state. If `None`, the initial 
            conditions are set to `[1, 0, ..., 0]` (all probability in the first state). If the `completeness` 
            attribute is `True`, the last state's probability is computed as `1 - sum(other_states)`, by default 
            this parameter is set as None.

        Returns
        -------
        dict[dict[str, float]]
            A dictionary containing the solution of the Markov Chain at the specified time:
            - 'by_states': A dictionary where keys are state names and values are the computed probabilities 
            for each state.
            - 'by_consequences': A dictionary where keys are consequence names and values are the summed 
            probabilities of states associated with each consequence. This is included only if `has_consequences` 
            is `True`.

        Raises
        ------
        ValueError
            If the 'time' parameter is not a float or an int.
        """

        if type(time) == int or type(time) == float:
            time = numpy.array([0., float(time)])
        else:
            raise ValueError("The 'time' parameter must be an int or float")

        if initial_conditions is None:
            if self.completeness:
                initial_conditions = numpy.array([1] + [0] * (self.size - 2))
            else:
                initial_conditions = numpy.array([1] + [0] * (self.size - 1))
        solution = odeintw.odeintw(self.__psys, initial_conditions, time, args=(self.matrix,))[-1]

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

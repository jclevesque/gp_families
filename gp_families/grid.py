
import numpy as np


class GridMap:
    '''
    Mapping of variables from ranges defined by min-max and scale to a
     0-1 unit hypercube.
    '''
    def __init__(self, variables):
        self.cardinality = 0

        # Count the total number of dimensions and roll into new format.
        for variable in variables:
            self.cardinality += 1
            if variable['type'] not in ['int', 'float', 'enum']:
                raise Exception("Unknown parameter type.")

        self.variables = variables
        print("Optimizing over %d dimensions\n" % (self.cardinality))

    def get_params(self, u):
        if u.shape[0] != self.cardinality:
            raise Exception("Hypercube dimensionality is incorrect.")

        params = []
        for variable, ui in zip(self.variables, u):
            if variable['type'] == 'int':
                val = variable['min'] + self._index_map(ui, variable['max']-variable['min']+1)
            elif variable['type'] == 'float':
                val = variable['min'] + ui*(variable['max']-variable['min'])
                #optional scale definition.
                scale = variable.get('scale', 'log')
                if scale == 'log':
                    val = 10**val
            elif variable['type'] == 'enum':
                ii = self._index_map(ui, len(variable['options']))
                val = variable['options'][ii]
            else:
                raise Exception("Unknown parameter type.")
            params.append({'name': variable['name'], 'val': val})
        return params

    def card(self):
        return self.cardinality

    def _index_map(self, u, items):
        u = np.max((u, 0.0))
        u = np.min((u, 1.0))
        return int(np.floor((1-np.finfo(float).eps) * u * float(items)))

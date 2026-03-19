import numpy as np
from .patterns import Patterns
import matplotlib.pyplot as plt
from .auxiliary_variable import AuxiliaryVariable
from .flow import Flow
from .stock import Stock
from .state import State

class Model:
    def __init__(self, patterns: Patterns = None):
        self._patterns = patterns if patterns is not None else Patterns(self)
        self.stocks = {}
        self.auxiliary_variables = {}

    def stock(self, name: str, initial_value):
        stock = Stock(name, initial_value, flows=[])
        self.stocks[name] = stock
        return stock

    def auxiliary_variable(self, name: str, operation: callable = None, *operands):
        auxiliary_variable = AuxiliaryVariable(name, operation=operation, operands=list(operands))
        self.auxiliary_variables[name] = auxiliary_variable
        return auxiliary_variable

    def flow(self, name: str, stock: Stock, operation: callable, *operands):
        flow = Flow(name, operation=operation, operands=list(operands))

        if flow not in stock.flows:
            stock.flows.append(flow)

        return flow

    def patterns(self, data):
        self._patterns.fit_sindy(data)

    def build(self):
        return State(stocks=list(self.stocks.values()), 
                     auxiliary_variables=list(self.auxiliary_variables.values()))

    def plot(self, history, title="Model simulation"):
        plt.figure(figsize=(10, 6))
        for name, values in history.items():
            arr = np.asarray(values)

            if arr.ndim == 1:
                plt.plot(arr, label=name)

            elif arr.ndim == 2:
                for i in range(arr.shape[1]):
                    plt.plot(arr[:, i], label=f"{name}_{i}")
        
        plt.title(title)
        plt.xlabel("Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

import numpy as np
from .auxiliary_variable import AuxiliaryVariable

class State:
    def __init__(self, stocks, auxiliary_variables):
        self.stocks = stocks if isinstance(stocks, list) else [stocks]
        self.auxiliary_variables = self._topological_order(
            auxiliary_variables if isinstance(auxiliary_variables, list) else [auxiliary_variables]
        )

    def _topological_order(self, aux_variables):
            ordered = []
            visited = set()
            def visit(v):
                if v not in visited:
                    visited.add(v)
                    if hasattr(v, 'operands'):
                        for op in v.operands:
                            if isinstance(op, AuxiliaryVariable): 
                                visit(op)
                    ordered.append(v)

            for aux_variable in aux_variables:
                visit(aux_variable)
            return ordered

    @property
    def flows(self):
        return [flow for stock in self.stocks for flow in stock.flows]

    def step(self, dt=1.0, attach_graph=False):
        for aux_variable in self.auxiliary_variables:
            aux_variable.update(attach_graph=attach_graph)
        
        for stock in self.stocks:
            for flow in stock.flows:
                flow.update(attach_graph=attach_graph)

        for stock in self.stocks:
            stock.integrate(dt, attach_graph=attach_graph)

    def simulate(self, steps, dt=1.0, attach_graph=False):
        history = {stock.name: [display_value(stock.value.value)] for stock in self.stocks}
        for aux_variable in self.auxiliary_variables:
            if "_operand_" not in aux_variable.name:
                history[aux_variable.name] = [display_value(aux_variable.value.value)]

        for _ in range(steps):
            self.step(dt, attach_graph=attach_graph)

            for stock in self.stocks:
                history[stock.name].append(display_value(stock.value.value))
            for aux_variable in self.auxiliary_variables:
                if "_operand_" not in aux_variable.name:
                    history[aux_variable.name].append(display_value(aux_variable.value.value))
         
        return history

    def optimize(self, steps, target_stock, parameters, mode, learning_rate=0.01, epochs=20):

        direction = 1.0 if mode=="maximize" else -1.0
        for epoch in range(epochs):
            for stock in self.stocks:
                stock.reset()

            self.simulate(steps=steps, attach_graph=True)

            target_stock.value.backprop()

            for parameter in parameters:
                #iadd MUST BE USED ONLY FOR OPTIMIZATION, NOT FOR EQUATIONS
                parameter.value += direction * learning_rate * parameter.gradient

            print(f"Epoch {epoch}: {target_stock.name} = {display_value(target_stock.value.value)}")


def display_value(value):
    if value is None:
        return 0.0
    array = np.asarray(value, dtype=np.float64)
    if array.size == 1:
        return float(array.reshape(-1)[0])
    return array.tolist()


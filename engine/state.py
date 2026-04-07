import numpy as np
from .auxiliary_variable import AuxiliaryVariable
from .element import Element

class State:
    def __init__(self, stocks, auxiliary_variables, dimensionality=1, randomness=0.0):
        self.dimensionality = dimensionality
        self.randomness = randomness
        self.stocks = stocks if isinstance(stocks, list) else [stocks]
        aux_list = auxiliary_variables if isinstance(auxiliary_variables, list) else [auxiliary_variables]
        if dimensionality > 1:
            for stock in self.stocks:
                base_value = stock.initial_value.value
                if randomness > 0:
                    # Apply randomness: normal distribution around base_value with std = base_value * randomness
                    vectorized_value = np.random.normal(base_value, abs(base_value) * randomness, dimensionality)
                else:
                    vectorized_value = np.ones(dimensionality) * base_value
                stock.initial_value = Element(vectorized_value, label=stock.name)
                stock.reset()
            for aux in aux_list:
                if aux.operation is None:
                    base_value = aux.value.value
                    if randomness > 0:
                        vectorized_value = np.random.normal(base_value, abs(base_value) * randomness, dimensionality)
                    else:
                        vectorized_value = np.ones(dimensionality) * base_value
                    aux.value = Element(vectorized_value, label=aux.name)
            for flow in [f for s in self.stocks for f in s.flows]:
                flow.update()
        self.auxiliary_variables = self._topological_order(aux_list)

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

        for stock in self.stocks:
            for flow in stock.flows:
                flow.update(attach_graph=attach_graph)

    def simulate(self, steps, dt=1.0, attach_graph=False, constraints: dict=None):
        # Initialize history with stocks
        history = {stock.name: [display_value(stock.value.value)] for stock in self.stocks}
        flows = {flow for stock in self.stocks for flow in stock.flows}

        # Initialize history with parameters (aux variables)
        for aux_variable in self.auxiliary_variables:
            if "_operand_" not in aux_variable.name:
                history[aux_variable.name] = [display_value(aux_variable.value.value)]

        for flow in flows:
            history[flow.name] = [display_value(flow.value.value)]

        for _ in range(steps):
            self.step(dt, attach_graph=attach_graph)

            if constraints:
                # Apply constraints to stocks
                for stock in self.stocks:
                    if stock.name in constraints:
                        boundaries = constraints[stock.name]
                        if "min" in boundaries:
                            stock.value.value = np.maximum(stock.value.value, boundaries["min"])
                        if "max" in boundaries:
                            stock.value.value = np.minimum(stock.value.value, boundaries["max"])

                # Apply constraints to auxiliary variables
                for aux_var in self.auxiliary_variables:
                    if aux_var.name in constraints:
                        boundaries = constraints[aux_var.name]
                        if "min" in boundaries:
                            aux_var.value.value = np.maximum(aux_var.value.value, boundaries["min"])
                        if "max" in boundaries:
                            aux_var.value.value = np.minimum(aux_var.value.value, boundaries["max"])

                # Apply constraints to flows
                
                for flow in flows:
                    if flow.name in constraints:
                        boundaries = constraints[flow.name]
                        if "min" in boundaries:
                            flow.value.value = np.maximum(flow.value.value, boundaries["min"])
                        if "max" in boundaries:
                            flow.value.value = np.minimum(flow.value.value, boundaries["max"])

            # Append stock values
            for stock in self.stocks:
                history[stock.name].append(display_value(stock.value.value))

            # Append parameter values
            for aux_variable in self.auxiliary_variables:
                if "_operand_" not in aux_variable.name:
                    history[aux_variable.name].append(display_value(aux_variable.value.value))

            # Append flow values
            for flow in flows:
                history[flow.name].append(display_value(flow.value.value))

        return history

    def optimize(self, steps, target_stock, parameters, mode, learning_rate=0.01, epochs=20, dt=1.0,
                 beta=0.9, beta_=0.999, epsilon=1e-8):

        for parameter in parameters:
            if parameter.operation is not None:
                raise ValueError(
                    f"Parameter '{parameter.name}' has an operation and cannot be optimized. "
                    f"Only leaf variables (constants/coefficients) are valid optimization targets."
                )

        direction = 1.0 if mode == "maximize" else -1.0

        # Adam optimizer state: first moment (mean) and second moment (variance) per parameter
        mean = [np.zeros_like(parameter.value.value) for parameter in parameters]
        variance = [np.zeros_like(parameter.value.value) for parameter in parameters]

        epoch_history = {target_stock.name: []}
        for parameter in parameters:
            epoch_history[parameter.name] = []

        for epoch in range(epochs):
            for stock in self.stocks:
                stock.reset()

            self.simulate(steps=steps, dt=dt, attach_graph=True)

            epoch_history[target_stock.name].append(display_value(target_stock.value.value))
            for parameter in parameters:
                epoch_history[parameter.name].append(display_value(parameter.value.value))

            target_stock.value.backprop()

            t = epoch + 1
            for index, parameter in enumerate(parameters):
                grad = direction * parameter.gradient

                # Update biased first and second moment estimates
                mean[index] = beta * mean[index] + (1 - beta) * grad
                variance[index] = beta_ * variance[index] + (1 - beta_) * grad ** 2

                # Bias-corrected estimates
                mean_hat = mean[index] / (1 - beta ** t)
                variance_hat = variance[index] / (1 - beta_ ** t)

                # iadd MUST BE USED ONLY FOR OPTIMIZATION, NOT FOR EQUATIONS
                parameter.value += learning_rate * mean_hat / (np.sqrt(variance_hat) + epsilon)
                parameter.value.gradient = np.zeros_like(parameter.value.gradient)

        return epoch_history


def display_value(value):
    if value is None:
        return 0.0
    array = np.asarray(value, dtype=np.float64)
    array = np.where(np.isfinite(array), array, 0.0)
    if array.size == 1:
        return float(array.reshape(-1)[0])
    return array.tolist()

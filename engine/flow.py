from .auxiliary_variable import AuxiliaryVariable


class Flow(AuxiliaryVariable):
    """Specific type of variable that represents a rate of change that affects the value of the Stock class"""
    # We call the initialization method of the Variable class to access the Flows attributes
    def __init__(self, name: str, operation: callable, operands):
        super().__init__(name=name, operation=operation, operands=operands)

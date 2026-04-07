from .element import Element

class AuxiliaryVariable:
    """In system dynamics, this is an AUXILIARY VARIABLE"""
    def __init__(self, name: str, operation: callable=None, operands=None):
        self.name = name
        self.operation = operation
        self.operands = operands if isinstance(operands, (list,tuple)) else [operands]

        if operation is None:
            if not self.operands or self.operands[0] is None:
                raise ValueError("AuxiliaryVariable requires at least one operand.")
            value = self.operands[0]
            self.value = value if isinstance(value, Element) else Element(value, label=self.name)
        else:
            self.operands = [
                operand if isinstance(operand, AuxiliaryVariable) or (
                    hasattr(operand, "value") and not isinstance(operand, Element)
                )
                else AuxiliaryVariable(name=f"{self.name}_coefficient_{index}", operands=[operand], operation=None)
                for index, operand in enumerate(self.operands)
            ]
            
            self.update()

    def update(self, operands=None, attach_graph=False):
        if self.operation is None:
            return self.value

        if operands is not None:
            _operands = operands if isinstance(operands, (list, tuple)) else [operands]
            self.operands = [
                operand if isinstance(operand, AuxiliaryVariable) or (
                    hasattr(operand, "value") and not isinstance(operand, Element)
                )
                else AuxiliaryVariable(name=f"{self.name}_coefficient_{index}", operands=[operand], operation=None)
                for index, operand in enumerate(_operands)
            ]
        
        values = [
            operand if isinstance(operand, Element) 
            else (operand.value if hasattr(operand, "value") else operand) 
            for operand in self.operands
        ]

        value = self.operation(*values)

        if attach_graph:
            self.value = value if isinstance(value, Element) else Element(value, label=self.name)
        else:
            raw_value = value.value if isinstance(value, Element) else value
            self.value = Element(raw_value, label=self.name)

    @property
    def gradient(self):
        return self.value.gradient

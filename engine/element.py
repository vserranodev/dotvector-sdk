import numpy as np
import matplotlib.pyplot as plt


class Element:
    def __init__(self, value, operands: tuple = (), operation: str = None, label=""):
        self.value = np.asarray(value, dtype=np.float64)
        self.operands = tuple(operands)
        self.operation = operation
        self.gradient = np.zeros_like(self.value, dtype=np.float64)
        self.backpropagation = lambda: None
        self.label = label


    def __add__(self, other):
        other = other if isinstance(other, Element) else Element(other)
        output = Element(
            self.value + other.value, 
            operands=(self, other), 
            operation="+", 
            label=f"{self.label} + {other.label}"
            )

        def backward():
            self.gradient += Element.shape_gradient(output.gradient, self.value.shape)
            other.gradient += Element.shape_gradient(output.gradient, other.value.shape)

        output.backpropagation = backward
        return output

    def __radd__(self, other):
        return self + other

    #iadd dunder method MUST BE USED ONLY FOR OPTIMIZATION, NOT FOR EQUATIONS, because it 
    # doesn't create a new Element object, it modifies the existing one. AND IF USED IN EQUATIONS
    # WE MIGHT LOSE TRACK OF THE COMPUTATIONAL GRAPH.
    
    def __iadd__(self, other):
        other_value = other.value if isinstance(other, Element) else other
        
        self.value += np.asarray(other_value, dtype=np.float64)
        
        self.operands = ()
        self.operation = None
        
        return self 

    def __mul__(self, other):
        other = other if isinstance(other, Element) else Element(other)
        output = Element(
            self.value * other.value,
            operands=(self, other),
            operation="*",
            label=f"{self.label} * {other.label}"
        )

        def backward():
            grad_self = other.value * output.gradient
            grad_other = self.value * output.gradient
            self.gradient += Element.shape_gradient(grad_self, self.value.shape)
            other.gradient += Element.shape_gradient(grad_other, other.value.shape)

        output.backpropagation = backward
        return output

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        other = other if isinstance(other, Element) else Element(other)
        output = Element(
            self.value - other.value,
            operands=(self, other),
            operation="-",
            label=f"{self.label} - {other.label}"
        )
        def backward():
            self.gradient += Element.shape_gradient(output.gradient, self.value.shape)
            other.gradient += Element.shape_gradient(-output.gradient, other.value.shape)

        output.backpropagation = backward
        return output

    def __rsub__(self, other):
        return Element(other) - self

    def __neg__(self):
        output = Element(-self.value, operands=(self,), operation="neg", label=f"-{self.label}")

        def backward():
            self.gradient += Element.shape_gradient(-output.gradient, self.value.shape)

        output.backpropagation = backward
        return output

    def __abs__(self):
        output = Element(np.abs(self.value), operands=(self,), operation="abs", label=f"abs({self.label})")

        def backward():
            grad_self = np.sign(self.value) * output.gradient
            self.gradient += Element.shape_gradient(grad_self, self.value.shape)

        output.backpropagation = backward
        return output

    def __truediv__(self, other):
        other = other if isinstance(other, Element) else Element(other)
        output = Element(
            self.value / other.value,
            operands=(self, other),
            operation="/",
            label=f"{self.label} / {other.label}"
        )

        def backward():
            grad_self = output.gradient / other.value
            grad_other = (-self.value / (other.value**2)) * output.gradient
            self.gradient += Element.shape_gradient(grad_self, self.value.shape)
            other.gradient += Element.shape_gradient(grad_other, other.value.shape)

        output.backpropagation = backward
        return output

    def __pow__(self, other):
        other = other if isinstance(other, Element) else Element(other)
        output = Element(
            self.value**other.value,
            operands=(self, other),
            operation="**",
            label=f"{self.label} ** {other.label}"
        )
        def backward():
            grad_self = other.value * (self.value ** (other.value - 1)) * output.gradient
            self.gradient += Element.shape_gradient(grad_self, self.value.shape)

            valid_base = self.value > 0
            other_grad = np.where(valid_base, output.value * np.log(self.value) * output.gradient, 0.0)
            other.gradient += Element.shape_gradient(other_grad, other.value.shape)

        output.backpropagation = backward
        return output

    # 
    def __bool__(self):
        """
        Why is this method necessary?
        In this SDK, comparison operators (==, >, <, etc.) do not return a standard 
        Python True/False. Instead, they return a new 'Element' object containing 
        a NumPy array of 1.0s and 0.0s. This is what allows logic to be 
        "differentiable" and tracked within the computational graph.
        
        The Problem:
        If when using the SDK we write 'if a == b:', Python internally calls __bool__ to decide 
        whether to enter the block. By default, Python would evaluate any existing 
        Element object as 'True', regardless of its internal values. This leads 
        to a slient and catastrophic bug in simulations.
        
        The Solution:
        We explicitly raise a ValueError. This forces the user to choose between:
        1. Python Logic: Using `(a == b).value.all()` to get a single boolean.
        2. Differentiable Logic: Using `(a == b)` as a multiplier in an equation 
           (e.g., `flow * (stock > 0)`).
        """
        raise ValueError(
            "The truth value of an Element is ambiguous. "
            "Use `(a == b).value.all()` for Python logic, "
            "or use it directly in mathematical equations for differentiable logic."
        )

    def log(self):
        value = np.log(np.maximum(self.value, 1e-15))
        output = Element(value, operands=(self,), operation="log", label=f"log({self.label})")

        def backward():
            gradient = (1.0 / np.maximum(self.value, 1e-15)) * output.gradient
            self.gradient += Element.shape_gradient(gradient, self.value.shape)

        output.backpropagation = backward
        return output

    def tanh(self):
        function = np.tanh(self.value)
        output = Element(function, operands=(self, ), label="tanh")

        def backward():
            grad_self = (1 - function**2) * output.gradient
            self.gradient += Element.shape_gradient(grad_self, self.value.shape)

        output.backpropagation = backward
        return output

    def __gt__(self, other):
        other = other if isinstance(other, Element) else Element(other)
        output = Element(
            (self.value > other.value).astype(np.float64), 
            operands=(self, other), 
            operation=">", 
            label=f"({self.label} > {other.label})"
        )
        return output

    def __lt__(self, other):
        other = other if isinstance(other, Element) else Element(other)
        output = Element(
            (self.value < other.value).astype(np.float64), 
            operands=(self, other), 
            operation="<", 
            label=f"({self.label} < {other.label})"
        )
        return output

    def __ge__(self, other):
        other = other if isinstance(other, Element) else Element(other)
        output = Element(
            (self.value >= other.value).astype(np.float64), 
            operands=(self, other), 
            operation=">=", 
            label=f"({self.label} >= {other.label})"
        )
        return output

    def __le__(self, other):
        other = other if isinstance(other, Element) else Element(other)
        output = Element(
            (self.value <= other.value).astype(np.float64), 
            operands=(self, other), 
            operation="<=", 
            label=f"({self.label} <= {other.label})"
        )
        return output

    def __eq__(self, other):
        other = other if isinstance(other, Element) else Element(other)
        output = Element(
            (self.value == other.value).astype(np.float64), 
            operands=(self, other), 
            operation="==", 
            label=f"({self.label} == {other.label})"
        )
        return output

    def __ne__(self, other):
        other = other if isinstance(other, Element) else Element(other)
        output = Element(
            (self.value != other.value).astype(np.float64), 
            operands=(self, other), 
            operation="!=", 
            label=f"({self.label} != {other.label})"
        )
        return output

    def backprop(self):
        graph = []
        visited = set()

        def computational_graph(node):
            node_id = id(node)
            if node_id not in visited:
                visited.add(node_id)
                for operand in node.operands:
                    computational_graph(operand)
                graph.append(node)

        computational_graph(self)
        for node in graph:
            node.gradient = np.zeros_like(node.value, dtype=np.float64)

        self.gradient = np.ones_like(self.value, dtype=np.float64)
        
        for node in reversed(graph):
            node.backpropagation()

    @staticmethod
    def shape_gradient(gradient, target_shape):
        """
        Reshape a backward gradient to match the original operand shape.

        Why is this function needed?
        - In forward pass, NumPy broadcasting can expand an operand.
        - In backward pass, the gradient arriving from the output may therefore
          have a larger shape than the original operand.
        - This method "undoes" that expansion by summing the broadcasted axes.

        Example:
        - Original operand shape: ()
        - Upstream gradient shape: (3,)
        - Result after shaping: () with value equal to the sum of 3 entries.
        """
        gradient = np.asarray(gradient, dtype=np.float64)

        # If gradient has extra leading Stocks, collapse them first.
        # This handles cases like grad=(2, 3, 4) for target=(3, 4).
        while gradient.ndim > len(target_shape):
            gradient = gradient.sum(axis=0)

        # 2) For axes where original target had size 1, sum that axis back.
        #    This is how we reverse NumPy broadcasting along singleton axes.
        for axis, size in enumerate(target_shape):
            if size == 1 and gradient.shape[axis] != 1:
                gradient = gradient.sum(axis=axis, keepdims=True)

        # 3) Return with the exact target shape expected by the operand.
        return gradient.reshape(target_shape)

    def __repr__(self):
        result = f"Label: {self.label}, Value: {self.value}"
        if self.operands:
            result += f", Operands: {self.operands}"
        if self.operation is not None:
            result += f", Operation: {self.operation}"
        return result

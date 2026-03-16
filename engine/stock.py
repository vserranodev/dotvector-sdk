try:
    from .element import Element
except ImportError:
    from element import Element

class Stock:
    """In system dynamics, this is a STOCK"""
    def __init__(self, name, value, flows):
        self.name = name
        self.initial_value = value if isinstance(value, Element) else Element(value, label=name)
        self.value = self.initial_value
        self.flows = flows if isinstance(flows, list) else ([flows] if flows else [])
        self.reset()

    def integrate(self, dt= 1.0, attach_graph=False):
        if not self.flows: 
            return 

        new_element = self.value + sum(flow.value * dt for flow in self.flows)

        if attach_graph:
            # Option to optimize, mmaintaining the full graph
            self.value = new_element
        else:
            # Option to simulate: We "reset" history saving only the number
            # We use .value to get a new Element without the history
            self.value = Element(new_element.value, label=self.name)

    def reset(self):
        """This function resets the Stock value to its 
        initial value to prepare it for a new simulation"""
        self.value = Element(self.initial_value.value.copy(), label=self.name)

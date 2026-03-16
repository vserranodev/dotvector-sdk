import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from engine.main import Model

if __name__ == "__main__":
    model = Model()
    stock = model.stock("stock", initial_value=100)
    state = model.build()
    history = state.simulate(steps=10)
    #model.plot(history)
    print(history)


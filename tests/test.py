
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from engine.main import Model
    import pandas as pd
    from engine.patterns import Patterns

    csv_path = Path(__file__).resolve().parents[1] / "tests" / "demo_data.csv"
    data = pd.read_csv(csv_path)

    model = Model()
    patterns = Patterns(model)
    patterns.fit_sindy(data)

    model.stock("stock", initial_value=100)
    state = model.build()
    history = state.simulate(steps=10)
    print(history)

    #model.plot(history)


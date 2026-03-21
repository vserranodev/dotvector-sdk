# DotVector

Differentiable programming and system dynamics modeling engine.

## Brief summary

DotVector is a framework that combines **automatic differentiation (autograd)** with **system dynamics** concepts. It allows building computational graphs, computing gradients via backpropagation, and modeling dynamic systems with stocks, flows and auxiliary variables.

* Core components

- **`Element`** — Computational-graph node with autograd support (scalar/vector/matrix values via NumPy). Thanks @karpathy for the inspiration. Supports arithmetic operations (`+`, `-`, `*`, `/`, `**`) with automatic gradient computation via backpropagation.
  Element is just the mathematical layer from which the computations of the model are made. The intention is to not have to interact with this class, just interact with the "Business" classes (Aux.Variable, Stock, Flows and State) in order to make the model declaration logical from a Business perspective.
- **`AuxiliaryVariable`** — Auxiliary variable in system dynamics. Wraps an operation over operands and updates dynamically.
- **`Stock`** — Stock in system dynamics. Represents an accumulated quantity that changes through flows.
- **`Flow`** — Flow in system dynamics. Represents a rate of change and can consume ordered operands (auxiliary variables or stocks).
- **`State`** — System state, made up of a set of stocks.

## Usage Flow

### 1) Learn from data

The recommended first step is to learn dynamic equations directly from historical data using `engine/patterns.py`.

How it works internally:

- You create stocks in `Model` with the same names as the numeric columns in your dataframe.
- `Patterns.fit_sindy(data)` trains a SINDy model from the input data.
- For each learned equation and each non-zero coefficient:
  - a persistent coefficient parameter is created as an `AuxiliaryVariable` with a descriptive `*_coefficient` name,
  - a flow expression is assembled from stock variables and those named coefficients.
- The learned flow operation is compatible with `Stock.integrate()`:
  - `stock_new = stock_old + flow * dt`.
- Each learned flow is attached to its target stock through `model.flow(...)`.

Minimal usage order:

1. Load dataframe and keep numeric columns.
2. Create one stock per numeric column.
3. Run `Patterns(model).fit_sindy(dataframe)` (or `model.patterns(dataframe)`).
4. Build and simulate with `state = model.build()` and `state.simulate(...)`.

## Project Structure

```text
dotvector/
├── engine/
│   ├── element.py  
│   ├── flow.py  
│   ├── stock.py   
│   ├── state.py  
│   ├── auxiliary_variable.py 
│   └── main.py
├── patterns/
│   └── patterns.py
├── tests/
│   └── test.py
└── README.md
```

## SINDy patterns module

The `patterns/` directory contains utilities to run pattern discovery with PySINDy.
It is intended for exploratory model identification workflows (e.g. financial or multivariate time-series inputs) that can later be translated into DotVector equations and flows.

## Technical Implementation

### 1) Overview

DotVector simulates a dynamic system (for example, customers and cash over time) and then reports which inputs have the strongest effect on the final outcome.

Example:

- If marketing spend increases, how much does final cash change?
- If churn increases, how much does final cash drop?

You do not calculate derivatives manually. The engine computes sensitivities automatically.

---

### 2) What each class represents

#### `Element` (the mathematical core)

Think of `Element` as a value plus memory of "how this value was produced."
It stores:

- a numeric value (`value`)
- references to parent nodes (`operands`) and the operation name (`operation`)
- a gradient (`gradient`) *(the derivative, the influence of an input to an output)*.

Every operation (`+`, `-`, `*`, `/`, `**`...) creates a new `Element` and stores a local backward rule.
This is what allows gradients *(the derivative, the influence of an input to an output)* to be propagated later with `backprop()`.

Concrete mini-example:

- `price = Element(50)`
- `customers = Element(100)`
- `revenue = customers * price`

Now `revenue` is not only a number. It also keeps the graph links to `customers` and `price`.
After `revenue.backprop()`, both `customers.gradient` and `price.gradient` are available.

Important implementation details:

- `value` is stored as a NumPy array (`np.asarray`), so it can represent a scalar, vector, or matrix. Initially it was stored as a scalar value, but in order to compute faster vectorized forms, we should add this layer of vectorization.
- `gradient` is initialized with `np.zeros_like(value)` (same shape as `value`).
- `backprop()` seeds the output with `np.ones_like(value)` and runs through the graph in reverse order. --> It goes from the output backwards and flows through the inputs.
- `shape_gradient(...)` fixes gradient shape after broadcasting.
- `display_value(...)` is a helper function in `backend/backend/engine.py` (outside the class) for API/serialization-safe output formatting.
- `Element` also supports `abs(x)` via `__abs__` and `x.log()` with a small numerical floor to avoid unstable `log(0)` paths.
- Comparison dunder operators (`>`, `<`, `>=`, `<=`, `==`, `!=`) currently return differentiable-compatible `Element` outputs containing 0/1 values for business-rule gating.

#### `AuxiliaryVariable` (auxiliary variable)

`AuxiliaryVariable` supports two runtime modes in the business model:

- **leaf variable**: fixed value, no operation (`operation=None`)
- **formula variable**: computed from ordered operands + operation

Examples:

- `acquisition = marketing_spend * conversion_rate`
- `gross_margin = price - variable_cost`

It does not accumulate by itself. It evaluates current output as an `Element`,
so its gradient *(the derivative, the influence of an input to an output)* can be traced automatically.

#### `Flow` (rate of change)

`Flow` extends `AuxiliaryVariable` and represents a rate that affects a stock (`Stock`).

Example:

- customer acquisition flow (inflow)
- customer churn flow (outflow)

A flow answers: "how much should this stock change per time step?"

Flow formulas consume ordered operands (auxiliary variables and/or stocks).

Concrete example:

- `acquisition = +marketing * conversion`
- `churn = -customers * churn_rate`

Both are formulas, but their sign defines whether they add to or subtract from the target stock.

#### `Stock` (stock / accumulated quantity)

In system dynamics, `Stock` is an accumulated quantity that changes through flows.

Examples:

- `Customers`
- `Cash`

Its `integrate(dt)` method updates the stock using:

- new value = current value + sum(flow * dt)

Concrete example:

- If `Customers = 100`, `acquisition = +8`, `churn = -3`, and `dt = 1`
- next `Customers = 100 + (8 - 3) * 1 = 105`

Stocks are where you usually read final KPIs and start backprop through `stock.value.backprop()`.

#### `State` (simulation orchestrator)

`State` groups:

- all stocks
- all active auxiliary variables/flows

In each `step(dt)`:

1. update variables and flows
2. integrate stocks with those updated flows

This is the execution engine for each simulation tick.

How gradients are used in practice:

1. Run simulation for N steps with `state.step()`.
2. Pick a target stock (for example `cash`).
3. Call `cash.value.backprop()`.
4. Read gradients *(the derivative, the influence of an input to an output)* on inputs such as
   `marketing_spend.gradient`, `price.gradient`, `churn_rate.gradient`.

---

### 3) End-to-end execution flow

This is the full runtime lifecycle, from model declaration to final sensitivity outputs.
The goal here is to explain it in plain words first, then connect each step to the exact function that runs in code.

### 0) Learn from data (Optional but recommended)

The recommended first step is to learn dynamic equations directly from historical data using the SINDy algorithm.

How it works:

- You create stocks in `Model` with the same names as the numeric columns in your dataframe (better to first rename the dataset columns for convenience)
- `Patterns.fit_sindy(data)` trains a SINDy model from the input data.
- For each learned equation:
  - a persistent coefficient parameter is created as an `AuxiliaryVariable`,
  - a flow expression is assembled from the equations the SINDy algorithm detected.
- The learned flow operation is compatible with `Stock.integrate()`:
  - `stock_new = stock_old + flow * dt`.
- Each learned flow is attached to its target stock through `model.flow(...)`.

#### Phase A: Model definition

1. Define base inputs as `AuxiliaryVariable` values (for example marketing spend, churn rate, price, fixed costs).

   - This means creating `AuxiliaryVariable(name, operands, operation=None)` for fixed inputs.
   - If `operation=None`, the variable behaves like an input parameter container.

   * *Concept:* AuxiliaryVariables are inputs, like Marketing Spend, Churn Rate, or Price.
   * *In code:* You create them as `AuxiliaryVariable(name, value)`. If they just hold a number, they act as simple inputs that you can manually tweak (or let the engine tweak later).
2. Defining system stocks with `Stock` (for example `customers`, `cash`) and their initial values.

   - In code, `Stock.__init__` stores both `initial_value` and `value`.
   - *Concept:* Stocks are things that accumulate over time. Think of your Bank Account (`Cash`) or your User Base (`Customers`).
   - *In code:* The declaration is made with a starting value (e.g., `Stock("Cash", 20000, flows=[])`)
3. Defining business dynamics as `Flow` objects.

   - A `Flow` is just an `AuxiliaryVariable` with "rate of change" meaning.
   - Each flow receives a callable + operands (for example `lambda m, c: m * c`).

   * *Concept:* Flows are the equations that move a stocks value.
   * *In code:* A `Flow` is just a formula. For example, `Acquisition = Marketing Spend * Conversion Rate`. Positive flows fill the "stock bucket"; negative flows drain it.
4. Attach each flow to the stock it updates.

   - Positive formulas increase the stock.
   - Negative formulas decrease the stock.
5. Build a `State` with:

   - all stocks (`Stocks`)
   - all active update units (`AuxiliaryVariables` and `Flows`)

#### Phase B: One simulation step (`state.step()`)

Every call to `step(dt)` executes two ordered stages:

1. **Update stage**: each variable/flow runs `update()`

   - Function used: `AuxiliaryVariable.update(...)`
   - What it does, step by step:
     1. Reads each operand.
        - If operand is already an `Element`, it uses it directly.
        - If operand is another object with `.value`, it uses that `.value`.
     2. Calls the variable formula (`self.operation(*values)`).
     3. Stores result as an `Element`.
   - Practical meaning:
     * The engine looks at every `AuxiliaryVariable` and `Flow` and calculates their current value based on the latest data.
     * *Example:* It calculates exactly how many users you will acquire *this specific month* based on the current marketing spend.
2. **Integration stage**: each stock runs `integrate(dt)`

   - Function used: `Stock.integrate(dt)`
   - What it does:

     1. Computes net flow: `sum(flow.value * dt for flow in self.flows)`
     2. Updates stock value: `self.value = self.value + net_flow`
   - Practical meaning:

     - the stock accumulates inflows/outflows for this time step.

     * The engine looks at the `Stocks` and applies their corresponding flows.
     * *The math:* **$new\_value = current\_value + \sum(flows \times dt)$**
     * *Example:* The `Cash` stock adds the new Revenue and subtracts the new Cost

Important technical points:

- Because from a granular layer values are `Element` objects, every arithmetic operation records graph dependencies.
- Over multiple steps, this builds a larger computation graph connecting final KPI values to earlier inputs.
- This graph is exactly what later enables gradients *(the derivative, the influence of an input to an output)*.

#### Phase C: Multi-step simulation loop

The state runs:

- `for _ in range(months): state.step()`

After this loop, each stock contains the final simulated state for the configured horizon.

#### Phase D: Choosing a target to optimize and running backpropagation to optimize that target

1. Select the output you care about (typically a stock value), for example:
   - `target = cash.value`
2. Run:
   - `target.backprop()`

What backprop does internally:

1. **Graph collection**
   - Function path: `Element.backprop()` -> `computational_graph(...)` (DFS traversal).
   - It builds the dependency map from the chosen target back to all inputs that contributed to that target.
   - Simple graph example:
     - `acquisition = marketing_spend * conversion_rate`
     - `revenue = customers * price`
     - `cash_next = cash_prev + revenue`
     - If target is `cash_next`, the graph includes `cash_next -> revenue -> (customers, price)` and the branch that produced `customers`, including `acquisition -> (marketing_spend, conversion_rate)`.
   - Why this step is needed:
     - Without this graph, the engine would not know which inputs actually influenced the target.
2. **Gradient reset**
   - Every node gradient becomes zero (`np.zeros_like(node.value)`).
   - Why reset is needed:
     - Gradients are accumulated with `+=`.
     - If you do not reset, old values from previous runs leak into the current run and results become incorrect.
     - Reset guarantees each backprop pass starts from a clean state.
3. **Target seed**
   - Target gradient is set to one (`np.ones_like(target.value)`).
   - Common-sense interpretation: "start influence propagation from the output itself".
   - Why seed is needed:
     - Backprop needs an initial "unit signal" at the output to start chain-rule propagation.
     - Setting target gradient to 1 means: "measure how every input changes this exact output."
4. **Reverse execution**
   - The graph is travelled in reverse order, from the outputs back through the inputs (`for node in reversed(graph)`).
   - Each node runs its local `backpropagation` function, which applies the derivative rule for its operation (`+`, `*`, `/`, `**`, etc.).
   - Why reverse order:
     - A node can only send gradient to its parents after it has received gradient from its children.
     - Reverse topological order guarantees the correct dependency order.
5. **Accumulation**
   - Upstream parameters receive gradients *(the derivative, the influence of an input to an output)*.
   - If broadcasting happened in forward pass, `Element.shape_gradient(...)` reshapes and sums correctly.
   - What "accumulation" means:
     - If one input affects the target through multiple paths, all path contributions are added.
     - Example: if `marketing_spend` influences both acquisition and marketing_costs, final `marketing_spend.gradient` is the sum of both effects.

Other interpretation:

- forward pass computes values
- backward pass distributes responsibility for the final result to every input

#### Phase E: Read and interpret derivatives/gradients/influences of each input to the output

After backprop, read gradients on controllable inputs.
These gradients *(the derivative, the influence of an input to an output)* quantify local sensitivity for the chosen target.
Examples:

- `dCash/dMarketing`
- `dCash/dPrice`
- `dCash/dChurn`

How to use them in practice:

- Sign tells direction (positive usually helps target, negative usually harms target).
- Magnitude tells local strength under current model assumptions.
- They are decision-support signals, not absolute truth forecasts.

---

### 4) What "gradient" means

A gradient is the **derivative, how much is something influenced by something. How an input contributes to an output.**

If `the derivative of the Cash with respect to the Marketing (dCash/dMarketing) = 0.64`, it means:

- around the current operating point, increasing marketing by 1 unit changes final cash by about 0.64 units.

It is not a perfect global forecast.
It is a local "influence score" inside your current model assumptions.

---

### 5) Scalars, arrays, and why `shape_gradient(...)` matters

DotVector supports:

- **scalar mode**: one value per variable
- **segmented mode**: vectors/matrices (for region, cohort, channel, scenario, etc.)

NumPy broadcasting allows operations like:

- `x = [1, 2, 3]`
- `b = 10`
- `y = x + b` -> `[11, 12, 13]`

In forward pass, `b` is expanded automatically.
In backward pass, gradients must return to the original operand shape.

Without `shape_gradient(...)`:

- you may try to add a vector gradient to a scalar gradient
- this causes shape mismatch or incorrect accumulation

With `shape_gradient(...)`:

- broadcasted axes are summed back
- each operand receives gradient with its original shape

Plain interpretation:

- if one scalar participates in 3 broadcasted outputs, its final gradient is the sum of those 3 contributions.

---

### 6) Practical business example

Suppose:

- marketing increases acquisition
- churn decreases customers
- customers and price drive revenue
- cash is affected by revenue and costs over multiple months

After simulation, `backprop()` on final cash gives:

- final KPI value
- contribution sensitivities from each controllable input

This supports decisions like:

- "Is increasing marketing worth it?"
- "Which hurts more right now: higher churn or lower price?"

---

### 7) Behavior of `State.optimize(...)`

`optimize(...)` is a simple gradient-based loop that updates selected parameters to improve a target stock.

Current signature in `backend/backend/engine.py`:

- `optimize(self, steps, target_stock, parameters, mode, learning_rate=0.01, epochs=20)`

What each argument means:

- `steps`: how many simulation steps are run inside each epoch
- `target_stock`: target stock object used as optimization goal
- `parameters`: list of parameter objects exposing `.value` and `.gradient` (for example auxiliary variables wrapping marketing/price)
- `mode`: optimization direction (`"maximize"` or any other string)
- `learning_rate`: update size
- `epochs`: number of optimization iterations

What happens internally, exactly:

1. It computes `direction`:
   - `+1.0` if `mode == "maximize"`
   - `-1.0` otherwise
2. For each epoch:
   - it resets every stock with `stock.reset()`
   - it runs the simulation horizon with `self.simulate(steps=steps)`
   - it runs `target_stock.value.backprop()`
   - it updates each parameter:
     - `parameter.value += direction * learning_rate * parameter.gradient`
   - it prints the current target value

Important implementation note:

- In `engine.py`, any `mode` different from `"maximize"` behaves as minimize (ternary branch).
- In the API layer, `OptimizationExecutionRequest.mode` is restricted to `"maximize" | "minimize"` by schema validation.
- The current implementation assumes `target_stock` exposes:
  - `name`
  - nested value access compatible with `target_stock.value.value` for reporting.

**FUTURE IMPLEMENTATIONS**:

- Hard comparison gates (`>`, `<`, etc.) are useful for business heuristics, but can create flat-gradient regions.
- If optimization quality around threshold rules becomes a bottleneck, introducing optional smooth gates (for example sigmoid-based approximations) in those critical equations can be a solution.

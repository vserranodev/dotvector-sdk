from pysindy import SINDy, STLSQ, PolynomialLibrary, SmoothedFiniteDifference
import ast
import re

class Patterns:
    def __init__(self, model):
        self.model = model

    def fit_sindy(
        self, 
        data, 
        epsilon=1e-2, 
        feature_library=PolynomialLibrary(degree=3),
        optimizer=STLSQ(threshold=1e-1, max_iter=20),
        differentiation_method=SmoothedFiniteDifference(),
        time_step=1
        ):
        dataframe = data.select_dtypes(include="number").replace([float("inf"), float("-inf")], float("nan")).dropna()
        columns = [column.replace(" ", "_") for column in dataframe.columns.tolist()]
        variables = columns
        x = dataframe.to_numpy(dtype=float)

        sindy = SINDy(
            optimizer=optimizer,
            feature_library=feature_library, 
            differentiation_method=differentiation_method,
            discrete_time=False)
        sindy.fit(x, t=time_step, feature_names=variables)
        
        score = sindy.score(x, t=time_step)
        if score < 0.8:
            return f"Model rejected by low precision (R² score: {score:.4f})"

        features = [feature.replace(" ", "*").replace("^", "**") for feature in sindy.get_feature_names()]
        stocks = [self.model.stocks[column] for column in columns]

        mapping = {f"x{index}": column_name for index, column_name in enumerate(columns)}

        for index, column in enumerate(columns):
            auxiliary_variables = []
            terms = []

            for coefficient, feature in zip(sindy.coefficients()[index], features):
                if abs(coefficient) < epsilon: 
                    continue
                # We assign the feature to a new variable to avoid modifying the original
                # item during the for loop

                _feature = feature
                for key, value in mapping.items():
                    _feature = re.sub(rf'\b{key}\b', value, _feature)

                feature_reformat = _feature.replace("**", "_pow_").replace("*", "_by_").replace(" ", "")
                
                if feature == "1":
                    variable_name = f"{column}_coefficient" 
                else:
                    variable_name = f"{feature_reformat}_coefficient" 

                auxiliary_variable = self.model.auxiliary_variable(variable_name, None, float(coefficient))
                auxiliary_variables.append(auxiliary_variable)
                                
                if feature == "1":
                    terms.append(auxiliary_variable.name)
                else:
                    terms.append(f"{auxiliary_variable.name} * {_feature}")

            if len(terms) == 0:
                raise ValueError(f"No features > epsilon")

            expression = " + ".join(terms).replace("+ -", "- ")
            code = compile(ast.parse(expression, mode="eval"), filename="<sindy>", mode="eval")

            # We compile the expression to a lambda function that evaluates the expression with the given arguments
            names = columns + [aux_variable.name for aux_variable in auxiliary_variables]

            # 2. La Lambda une todos los nombres con todos los valores (args)
            operation = lambda *args, code=code, names=names: (
                eval(
                    code,
                    {"__builtins__": {}},
                    dict(zip(names, args)) 
                )
            )
            self.model.flow(
                name=f"{column}_flow",
                stock=self.model.stocks[column],
                operation=operation,
                operands = stocks + auxiliary_variables,
            )


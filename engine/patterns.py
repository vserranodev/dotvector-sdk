from pysindy import SINDy, STLSQ, PolynomialLibrary, SmoothedFiniteDifference
import ast
import re

class Patterns:
    def __init__(self, model):
        self.model = model

    def fit_sindy(self, data, threshold=1e-1, epsilon=1e-2):
        dataframe = data.select_dtypes(include="number").replace([float("inf"), float("-inf")], float("nan")).dropna()
        columns = [column.replace(" ", "_") for column in dataframe.columns.tolist()]
        variables = columns
        x = dataframe.to_numpy(dtype=float)

        sindy = SINDy(
            optimizer=STLSQ(threshold=threshold),
            feature_library=PolynomialLibrary(degree=3), 
            differentiation_method=SmoothedFiniteDifference(),
            discrete_time=False)
        sindy.fit(x, t=1)
        

        if sindy.score(x) < 0.8:
            return f"Model rejected by low precision (R²: {sindy.score(x):.4f})"

        features = [feature.replace(" ", "*").replace("^", "**") for feature in sindy.get_feature_names(input_features=variables)]
        stocks = [self.model.stocks[column] for column in columns]

        mapping = {f"x{index}": column_name for index, column_name in enumerate(columns)}

        for index, column in enumerate(columns):
            aux_variables = []
            terms = []

            for coefficient, feature in zip(sindy.coefficients()[index], features):
                if abs(coefficient) < epsilon: 
                    continue
                
                feature_copy = feature
                for key, value in mapping.items():
                    feature_copy = re.sub(rf'\b{key}\b', value, feature_copy)

                format_feature_copy = feature_copy.replace("**", "_pow_").replace("*", "_by_").replace(" ", "")
                
                if feature == "1":
                    variable_name = f"{column}_parameter" 
                else:
                    variable_name = f"weight_{column}_{format_feature_copy}" 

                auxiliary_variable = self.model.auxiliary_variable(variable_name, None, float(coefficient))
                aux_variables.append(auxiliary_variable)
                
                aux_var_name = auxiliary_variable.name 
                
                if feature == "1":
                    terms.append(aux_var_name)
                else:
                    terms.append(f"{aux_var_name} * {feature_copy}")

            if len(terms) == 0:
                raise ValueError(f"No features > epsilon")

            expression = " + ".join(terms).replace("+ -", "- ")
            code = compile(ast.parse(expression, mode="eval"), "<sindy>", "eval")

            # We compile the expression to a lambda function that evaluates the expression with the given arguments
            operation = lambda *args, code=code, columns=columns, n_stocks=len(columns): (
                eval(
                    code,
                    {"__builtins__": {}},
                    dict(
                        zip(columns, args[:n_stocks]),
                        **{f"parameter_{key}": value for key, value in enumerate(args[n_stocks:])}
                    )
                )
            )

            self.model.flow(
                name=f"{column}_flow",
                stock=self.model.stocks[column],
                operation=operation,
                operands=list(stocks + aux_variables),
            )


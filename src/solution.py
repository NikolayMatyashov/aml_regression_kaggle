import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

from src import data_analyzing


def predict_solution(X_train, y_train, X_test):
    """
    This method uses prepared data and AdaBoost including Decision tree to generate
    predictions from the given test set.
    """

    # Normalize data
    scaler = MinMaxScaler()
    scaler.fit(X_train.values)
    X_train = scaler.transform(X_train.values)
    X_test = scaler.transform(X_test.values)

    # Build and fit chosen model
    model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=30, criterion='friedman_mse'),
                              n_estimators=50, learning_rate=0.5, loss='square')
    model.fit(X_train, y_train)

    return model.predict(X_test)


def test_solution():
    X_train, X_test, Y_train, Y_test = data_analyzing.get_train_data(with_test_split=True)
    Y_hat = predict_solution(X_train, Y_train, X_test)
    print("MAE: %f" % np.mean(abs(Y_hat - Y_test)))

def generate_solution():
    X_train, Y_train = data_analyzing.get_train_data()
    X_test = data_analyzing.get_test_data()

    Y_hat = predict_solution(X_train, Y_train, X_test)

    # Create csv solution
    solution = pd.DataFrame(Y_hat, columns=['price'], index=X_test.index)
    solution.to_csv(r'../data/solution.csv')


if __name__ == '__main__':
    test_solution()
    generate_solution()

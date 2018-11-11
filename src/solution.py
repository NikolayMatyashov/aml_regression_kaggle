import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

from src import data_analyzing


def generate_solution():
    """
    This method uses prepared data and AdaBoost including Decision tree to generate
    predictions from the given test set and saves them to 'data/solution.csv'.
    """

    # Get data
    X_prepared, train_Y = data_analyzing.get_data()
    X_test_prepared = data_analyzing.get_test_data()

    # Normalize data
    scaler = MinMaxScaler()
    scaler.fit(X_prepared.values)
    index = X_test_prepared.index  # Save indexes for X in test
    X_test = scaler.transform(X_test_prepared.values)
    X_train = scaler.transform(X_prepared.values)

    # Build and fit chosen model
    model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=30, criterion='friedman_mse'),
                              n_estimators=50, learning_rate=0.5, loss='square')
    model.fit(X_train, train_Y)
    Y_hat = model.predict(X_test)

    # Create csv solution
    solution = pd.DataFrame(Y_hat, columns=['price'], index=index)
    solution.to_csv('data/solution.csv')


if __name__ == '__main__':
    generate_solution()

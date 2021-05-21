"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data, process_data_combine
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)
    return vs, mape, mae, mse, math.sqrt(mse), r2


def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    # d = '2016-3-4 00:00'
    # x = pd.date_range(d, periods=288, freq='5min')

    x = np.array(range(288))
    time_list = ['00:00', '02:00', '04:00', '06:00', '08:00',
                 '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00']
    aa = np.linspace(0, 288, 12)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    # ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)

    plt.xticks(aa, time_list, rotation=0)

    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    fig.autofmt_xdate()

    plt.show()


def main():
    lstm = load_model('model/lstm.h5')
    # lstm = load_model('model/lstm-no-drop.h5')
    gru = load_model('model/gru.h5')
    saes = load_model('model/saes.h5')
    models = [lstm, gru, saes]
    names = ['LSTM', 'gru', 'SAEs']

    # # 12 24 46尺度的LSTM
    # lstm12 = load_model('model/lstm_12.h5')
    # lstm24 = load_model('model/lstm_24.h5')
    # lstm36 = load_model('model/lstm_36.h5')

    # models = [lstm12, lstm24, lstm36]
    # names = ['LSTM_12', 'LSTM_24' ,'LSTM_36']

    lag = 24
    file1 = 'data/train.csv'
    file2 = 'data/test.csv'
    _, _, X_test12, y_test12, _ = process_data_combine(file1, file2, 12, 0)
    _, _, X_test24, y_test24, scaler = process_data_combine(file1, file2, 24, 0)
    _, _, X_test36, y_test36, _ = process_data_combine(file1, file2, 36, 0)

    num_lane = len(X_test12)

    X_test = X_test24
    y_test = y_test24

    for a in range(1):
        # 融合模型
        evs, mapes, maes, mses, rmses, r2s = np.zeros(num_lane), np.zeros(num_lane), np.zeros(num_lane), \
                                             np.zeros(num_lane), np.zeros(num_lane), \
                                             np.zeros(num_lane)  # explained_variance_score
        # y_preds = []
        for j in range(num_lane):
            stack_predict = []
            # stack_predict = np.zeros(288)

            for name, model in zip(names, models):

                if name == 'SAEs':
                    X_test[j] = np.reshape(X_test[j], (X_test[j].shape[0], X_test[j].shape[1]))
                else:
                    X_test[j] = np.reshape(X_test[j], (X_test[j].shape[0], X_test[j].shape[1], 1))

                predicted = model.predict(X_test[j])
                predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
                stack_predict.append(predicted)
                # y_preds.append(predicted[:288])
                # print(name)
            predicted = (7 * stack_predict[0] + 0 * stack_predict[1] + 3 * stack_predict[2]) / 10
            # predicted = (10 * stack_predict[0] + 0 * stack_predict[1] + 0 * stack_predict[2]) / 10
            y_test[j] = scaler.inverse_transform(y_test[j].reshape(-1, 1)).reshape(1, -1)[0]
            stack_predict.append(predicted)
            stack_predict.pop(1)

            name2 = ['LSTM', 'SAEs', 'LSTM-SAEs']
            plot_results(y_test[: 288], stack_predict, name2)

            vs, mape, mae, mse, rmse, r2 = eva_regress(y_test[j], predicted)
            evs[j] = vs
            mapes[j] = mape
            maes[j] = mae
            mses[j] = mse
            rmses[j] = rmse
            r2s[j] = r2
            # print("完成了" + str(j))

        print(str(a), str(10 - a), evs.mean(), mapes.mean(), maes.mean(), mses.mean(), rmses.mean(), r2s.mean(),
              r2s.mean() / evs.mean())


if __name__ == '__main__':
    main()

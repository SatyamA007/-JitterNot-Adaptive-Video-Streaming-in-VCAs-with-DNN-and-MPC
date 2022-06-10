import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

csv_dir = 'Features'
ct = 100


def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x
    #
    # # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    print(b_0, b_1)

    return b_0, b_1


def get_fps_x_y(csv_path):
    df = pd.read_csv(csv_path)
    # print(df.head())
    tp = 'outbound_packetsSent/s'
    frames_sent_per_sec = 'outbound_framesSent/s'
    resolution = 'resolution'
    df[resolution] = df['outbound_frameWidth'] * df['outbound_frameHeight']
    # TODO : how to deal with infinite values ??
    # df = df[df[tp] > 0]
    df = df[df[tp] > 0]
    df['t/r'] = df[resolution] / df[tp]

    print(df['t/r'])
    x = list(df['t/r'])
    y = list(df[frames_sent_per_sec])
    return x, y


def get_qps(csv_path):
    df = pd.read_csv(csv_path)
    tp = 'outbound_packetsSent/s'
    qpsum = 'outbound_qpSum/framesEncoded'
    fps = 'outbound_framesPerSecond'
    rt = 'tp/fps'
    label = 'label'
    df = df[df[fps] > 0]
    df[rt] = df[tp]  * 100 / (df['outbound_frameWidth'] * df['outbound_frameHeight'] )
    df = df[df[qpsum] > 0]
    df[label] = df[qpsum] * df[fps]
    x = list(df[rt])
    y = list(df[label])
    return x, y




x_global = []
y_global = []
m_global = []
n_global = []
print(os.curdir)
for dir in os.listdir(csv_dir):
    if os.path.isdir(os.path.join(csv_dir, dir)):
        for csv_file in os.listdir(os.path.join(csv_dir, dir)):
            print("*****************************************************")
            print(csv_file)
            if 'DS_Store' in csv_file:
                continue
            ct -= 1
            if ct < 0:
                break
            csv_path = os.path.join(csv_dir, dir, csv_file)
            # p, q = get_jitter_x_y(csv_path)
            p, q, op, oq = get_qps(csv_path)
            x_global.extend(p)
            y_global.extend(q)


def plot_regression(x, y):
    # Importing Linear Regression
    from sklearn.linear_model import LinearRegression
    # importing libraries for polynomial transform
    from sklearn.preprocessing import PolynomialFeatures
    # for creating pipeline
    from sklearn.pipeline import Pipeline
    # creating pipeline and fitting it on data
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=15)
    plt.scatter(x_global,y_global, s=15, color='darkturquoise')
    # for degree in range(5,6):
    degree = 3

    myline = np.linspace(0, 0.2, 120)
    mymodel = np.poly1d(np.polyfit(x, y, degree))
    print(mymodel.coeffs)

    plt.plot(myline, mymodel(myline), color='red', label='Polynomial Regression')
    plt.xlabel('Throughput/Resolution (in hundredths)', fontsize=16)
    plt.ylabel('qpSum/s', fontsize=16)
    plt.legend()
    plt.show()

# qps = 'r/c'
x, y = np.asarray(x_global), np.asarray(y_global)
plot_regression(x, y)


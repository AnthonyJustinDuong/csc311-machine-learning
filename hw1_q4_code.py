import numpy as np
from matplotlib import pyplot
data_train = {'X': np.genfromtxt('data/data_train_X.csv', delimiter=','),
    't': np.genfromtxt('data/data_train_y.csv', delimiter=',')}
data_test = {'X': np.genfromtxt('data/data_test_X.csv', delimiter=','),
    't': np.genfromtxt('data/data_test_y.csv', delimiter=',')}

def shuffle_data(data):
    t, X = data
    perm = np.random.permutation(len(t))
    return t[perm], X[perm]

def split_data(data, num_fold, fold):
    t, X = data
    t_rest = np.array_split(t, num_fold)
    X_rest = np.array_split(X, num_fold)
    t_fold = t_rest.pop(fold - 1)
    X_fold = X_rest.pop(fold - 1)
    return (t_fold, X_fold), (np.concatenate(t_rest), np.concatenate(X_rest))

def train_model(data, lambd):
    t, X = data
    to_invert = np.transpose(X) @ X + lambd * X.shape[0] * np.identity(X.shape[1])
    return np.linalg.inv(to_invert) @ np.transpose(X) @ t

def predict(data, model):
    t, X = data
    return X @ model

def loss(data, model):
    t, X = data
    diff = predict(data, model) - t
    return np.dot(diff, diff) / (2 * X.shape[0])

def cross_validation(data, num_folds, lambd_seq):
    data = shuffle_data(data)
    cv_error = []
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(1, num_folds + 1):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error.append(cv_loss_lmd / num_folds)
    return cv_error

if __name__ == '__main__':
    # Set up the lambda sequence
    lambd_len = 50
    lambd_start = 0.00005
    lambd_end = 0.005
    interval_len = lambd_end - lambd_start
    lambd_seq = np.arange(lambd_start, lambd_end, interval_len / lambd_len)

    # Part (c)
    training_errs = []
    test_errs = []
    d_train = (data_train['t'], data_train['X'])
    d_test = (data_test['t'], data_test['X'])
    for lambd in lambd_seq:
        model = train_model(d_train, lambd)
        training_errs.append(loss(d_train, model))
        test_errs.append(loss(d_test, model))

    # Part (d)
    cv5 = cross_validation(d_train, 5, lambd_seq)
    cvA = cross_validation(d_train, 10, lambd_seq)

    cv5_best_lambd = lambd_seq[np.argmin(cv5)]
    cvA_best_lambd = lambd_seq[np.argmin(cvA)]
    print("Proposed lambda for 5-fold cross validation = {l}:".format(l=cv5_best_lambd))
    print("Proposed lambda for 10-fold cross validation = {l}:".format(l=cvA_best_lambd))

    # Plot errors
    pyplot.plot(lambd_seq, training_errs, 'o-', label='Training')
    pyplot.plot(lambd_seq, test_errs, 'o-', label='Test')
    pyplot.plot(lambd_seq, cv5, 'o-', label='5-fold CV')
    pyplot.plot(lambd_seq, cvA, 'o-', label='10-fold CV')

    # Title and label plot
    pyplot.title('Training, Test, and Cross-Validation Errors for Ridge Regression')
    pyplot.xlabel('lambda')
    pyplot.ylabel('error')

    pyplot.legend()
    pyplot.show()

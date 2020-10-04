from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy
from matplotlib import pyplot

def load_data(fake_data_path, real_data_path):
    y = []

    # Convert clean_fake.txt to a list of strings
    fake_headlines = []
    with open(fake_data_path, "r") as file:
        for line in file:
          fake_headlines.append(line.strip())
          y.append(0)

    # Convert clean_real.txt to a list of strings
    real_headlines = []
    with open(real_data_path, "r") as file:
        for line in file:
          real_headlines.append(line.strip())
          y.append(1)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(fake_headlines + real_headlines)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)
    return X_train, X_val, X_test, y_train, y_val, y_test

def select_knn_model(X_train, X_val, y_train, y_val):
    k_range = range(1,21)
    training_accs = []
    validation_accs = []

    # Gather training and validation accuracies for values of k
    for n in k_range:
        neigh = KNeighborsClassifier(n_neighbors=n, metric='cosine')
        neigh.fit(X_train, y_train)
        training_accs.append(neigh.score(X_train, y_train))
        validation_accs.append(neigh.score(X_val, y_val))

    # Plot accuracies
    pyplot.plot(k_range, training_accs, 'o-', label='Training')
    pyplot.plot(k_range, validation_accs, 'o-', label='Validation')

    # Title and label plot
    pyplot.title('Training and Test Accuracy of KNN with headline data')
    pyplot.xlabel('k = Number of Nearest Neighbours')
    pyplot.ylabel('accuracy = # of correct / # of total')

    pyplot.legend()
    pyplot.show()

    # Find k with best validation accuracy
    best_k = numpy.argmax(validation_accs) + 1
    return best_k

if __name__ == '__main__':
    local_fake_data_path = "data/clean_fake.txt"
    local_real_data_path = "data/clean_real.txt"
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(local_fake_data_path, local_real_data_path)
    best_k = select_knn_model(X_train, X_val, y_train, y_val)

    # Report the test accuracy of model with best k
    neigh = KNeighborsClassifier(n_neighbors=best_k, metric='cosine')
    neigh.fit(X_train, y_train)
    print("Test Accuracy for k = {k}:".format(k=best_k))
    print(neigh.score(X_test, y_test))

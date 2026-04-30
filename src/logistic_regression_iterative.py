import time
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b

import time

def propagate(w, b, X, Y):
    # m é o número de exemplos
    m = X.shape[1] # Número de exemplos
    num_features = X.shape[0]

    cost = 0
    dw = np.zeros((num_features, 1))
    db = 0

    A_values = np.zeros((1, m))

    for i in range(m):
        # Forward Propagation
        z_i = np.dot(w.T, X[:, i].reshape(-1, 1)) + b
        a_i = sigmoid(z_i)

        A_values[0, i] = a_i.item()

        # Cálculo do Custo
        cost += Y[0, i] * np.log(a_i) + (1 - Y[0, i]) * np.log(1 - a_i)

        # Backward Propagation
        dz_i = a_i - Y[0, i]
        dw += X[:, i].reshape(-1, 1) * dz_i
        db += dz_i

    cost = -1/m * cost
    dw = dw / m
    db = db / m

    cost = np.squeeze(cost) # Garante que o custo seja um escalar

    grads = {"dw": dw, "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):

    costs = []
    start_optimization_time = time.time()

    for i in range(num_iterations):
        # Calcula o custo e o gradiente usando a função de propagação não vetorizada
        grads, cost = propagate(w, b, X, Y)

        # Recupera os gradientes
        dw = grads["dw"]
        db = grads["db"]

        # Atualiza os parâmetros
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Registra o custo
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print (f"Custo após iteração {i}: {cost:.4f}")

    total_optimization_time = time.time() - start_optimization_time
    params = {"w": w, "b": b}

    return params, costs, total_optimization_time

def predict(w, b, X):

    # Faz a previsão final: se a probabilidade for > 0.5, classifica como 1 (spam)
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    for i in range(m):
        z_i = np.dot(w.T, X[:, i].reshape(-1, 1)) + b
        a_i = sigmoid(z_i)
        Y_prediction[0, i] = 1 if a_i > 0.5 else 0

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.01, print_cost=False):
    # Função principal que junta inicialização, otimização e predição
    w, b = initialize_with_zeros(X_train.shape[0])
    params, costs, total_time = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Calcula acurácia comparando as previsões com os rótulos reais
    Y_prediction_test = predict(params["w"], params["b"], X_test)
    Y_prediction_train = predict(params["w"], params["b"], X_train)

    train_acc = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_acc = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100


    return {"costs": costs, "train_acc": train_acc, "test_acc": test_acc, "total_optimization_time": total_time}

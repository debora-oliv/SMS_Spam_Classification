import numpy as np
import time

def sigmoid(z):
    # Função de ativação que mapeia qualquer valor para o intervalo entre 0 e 1
    return 1 / (1 + np.exp(-z))

def initialize_with_zeros(dim):
    # Cria pesos (w) como zeros e viés (b) como zero para começar o treino
    w = np.zeros((dim, 1))
    b = 0
    return w, b

# Váriavel que armazena custo para fins de comparação entre implementações
v_cost_final = 0.0

def propagate(w, b, X, Y):
    # m é o número de exemplos
    m = X.shape[1]

    # Forward Propagation: calcula a previsão atual
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)

    # Calcula o 'Custo' (erro) usando Log Loss
    cost = -1/m * (np.sum(Y*np.log(A) + (1 - Y)*np.log(1 - A)))

    # Backward Propagation: calcula o gradiente (direção para ajustar os pesos)
    dw = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * np.sum(A - Y)

    return {"dw": dw, "db": db}, np.squeeze(cost)

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    start_optimization_time = time.time()

    # Loop de treinamento (Gradiente Descendente)
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        # Atualiza os pesos e o viés subtraindo o gradiente * taxa de aprendizado
        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]

        if i % 100 == 0:
            costs.append(cost)

    return {"w": w, "b": b}, costs, time.time() - start_optimization_time

def predict(w, b, X):
    # Faz a previsão final: se a probabilidade for > 0.5, classifica como 1 (spam)
    m = X.shape[1]
    A = sigmoid(np.dot(w.reshape(X.shape[0], 1).T, X) + b)
    Y_prediction = np.zeros((1, m))
    Y_prediction[A > 0.5] = 1
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
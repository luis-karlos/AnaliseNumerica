import numpy as np

def metodo_euler_explicito(func, condicao_inicial, T, dt):
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]
    d = len(y0)
    n = int((T - t0) / dt)

    t_out = np.zeros(n + 1)
    y_out = np.zeros((n + 1, d))

    # Define o primeiro ponto da solução (a condição inicial)
    t_out[0] = t0
    y_out[0] = y0

    for i in range(n):
        # Pega os valores do passo anterior (usando o índice i)
        ti = t_out[i]
        yi = y_out[i]

        # A fórmula do método de Euler explícito
        y_out[i + 1] = yi + dt*func(ti, yi)

        # Atualiza o vetor de tempo
        t_out[i + 1] = ti + dt

    return t_out, y_out

def metodo_rk4(func, condicao_inicial, T, dt):
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]
    d = len(y0)
    n = int((T - t0) / dt)
    
    t_out = np.zeros(n + 1)
    y_out = np.zeros((n + 1, d))

    # Define o primeiro ponto da solução (a condição inicial)
    t_out[0] = t0
    y_out[0] = y0

    A = np.array([
        [0,   0,   0, 0],
        [0.5, 0,   0, 0],
        [0,   0.5, 0, 0],
        [0,   0,   1, 0]
    ])
    B = np.array([1/6, 1/3, 1/3, 1/6])
    C = np.array([0, 0.5, 0.5, 1])

    for i in range(n):
        # Pega os valores do passo anterior (usando o índice i)
        ti = t_out[i]
        yi = y_out[i]

        f1 = dt * func(ti, yi)
        f2 = dt*func(ti + C[1]*dt, yi + A[1,0]*f1)
        f3 = dt*func(ti + C[2]*dt, yi + A[2,1]*f2)
        f4 = dt*func(ti + C[3]*dt, yi + A[3,2]*f3)

        y_out[i + 1] = yi + B[0]*f1 + B[1]*f2 + B[2]*f3 + B[3]*f4
        t_out[i + 1] = ti + dt

    return t_out, y_out

def metodo_euler_explicito_linear(Acoef, condicao_inicial, T, dt):
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]
    d = Acoef.shape[0]
    n = int((T - t0) / dt)

    t_out = np.zeros(n + 1)
    y_out = np.zeros((n + 1, d))

    # Define o primeiro ponto da solução (a condição inicial)
    t_out[0] = t0
    y_out[0] = y0

    for i in range(n):
        # Pega os valores do passo anterior (usando o índice i)
        ti = t_out[i]
        yi = y_out[i]

        # A fórmula do método de Euler explícito
        y_out[i + 1] = yi + dt*(Acoef@yi)

        # Atualiza o vetor de tempo
        t_out[i + 1] = ti + dt

    return t_out, y_out

def metodo_rk4_linear(Acoef, condicao_inicial, T, dt):
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]
    d = Acoef.shape[0]
    n = int((T - t0) / dt)
    
    t_out = np.zeros(n + 1)
    y_out = np.zeros((n + 1, d))

    # Define o primeiro ponto da solução (a condição inicial)
    t_out[0] = t0
    y_out[0] = y0

    A = np.array([
        [0,   0,   0, 0],
        [0.5, 0,   0, 0],
        [0,   0.5, 0, 0],
        [0,   0,   1, 0]
    ])
    B = np.array([1/6, 1/3, 1/3, 1/6])
    C = np.array([0, 0.5, 0.5, 1])

    for i in range(n):
        # Pega os valores do passo anterior (usando o índice i)
        ti = t_out[i]
        yi = y_out[i]

        f1 = dt * Acoef@yi
        f2 = dt*Acoef@(yi + A[1,0]*f1)
        f3 = dt*Acoef@(yi + A[2,1]*f2)
        f4 = dt*Acoef@(yi + A[3,2]*f3)

        y_out[i + 1] = yi + B[0]*f1 + B[1]*f2 + B[2]*f3 + B[3]*f4
        t_out[i + 1] = ti + dt

    return t_out, y_out
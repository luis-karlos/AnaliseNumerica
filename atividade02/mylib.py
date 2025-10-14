import numpy as np
import math

def erro_relativo(y_exato, y_aproximado):
    """
    Calcula o erro relativo entre a solução exata e a aproximada.

    Parameters
    ----------
    y_exato : array-like
        Solução exata.
    y_aproximado : array-like
        Solução aproximada.

    Returns
    -------
    array-like
        Erro relativo.
    """
    return np.abs((y_exato - y_aproximado) / y_exato)

def erro_local(y_exato, y_aproximado):
    """
    Calcula o erro local entre a solução exata e a aproximada.

    Parameters
    ----------
    y_exato : array-like
        Solução exata.
    y_aproximado : array-like
        Solução aproximada.

    Returns
    -------
    array-like
        Erro local.
    """
    return np.abs(y_exato - y_aproximado)

def metodo_euler_explicito(func, condicao_inicial, T, dt):
    """
    Resolve um PVI usando o método de Euler explícito.

    Parameters
    ----------
    func : callable
        Função f(t, x) do problema.
    condicao_inicial : list or tuple
        [t0, x0] condição inicial.
    T : float
        Tempo final.
    dt : float
        Passo de integração.

    Returns
    -------
    t_out : ndarray
        Vetor de tempos.
    y_out : ndarray
        Vetor de soluções aproximadas.
    """
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]

    n = int((T - t0) / dt)

    t_out = np.zeros(n + 1)
    y_out = np.zeros(n + 1)

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

def metodo_serie_taylor(func, condicao_inicial, T, dt):
    """
    Resolve um PVI usando o método da série de Taylor de ordem arbitrária.

    Parameters
    ----------
    func : list of callables
        Lista de funções das derivadas [f0, f1, ..., fn].
    condicao_inicial : list or tuple
        [t0, x0] condição inicial.
    T : float
        Tempo final.
    dt : float
        Passo de integração.

    Returns
    -------
    t_out : ndarray
        Vetor de tempos.
    y_out : ndarray
        Vetor de soluções aproximadas.
    """
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]

    n = int((T - t0) / dt)
    ordem = len(func)

    t_out = np.zeros(n + 1)
    y_out = np.zeros(n + 1)

    # Define o primeiro ponto da solução (a condição inicial)
    t_out[0] = t0
    y_out[0] = y0

    for i in range(n):
        # Pega os valores do passo anterior (usando o índice i)
        ti = t_out[i]
        yi = y_out[i]

        soma_taylor = 0.0

        for k in range(ordem):
            # Termo (k+1) da série de Taylor
            valor_derivada = func[k](ti, yi)
            fatorial = math.factorial(k + 1)
            soma_taylor += (dt**(k + 1) / fatorial) * valor_derivada

        y_out[i + 1] = yi + soma_taylor

        # Atualiza o vetor de tempo
        t_out[i + 1] = ti + dt

    return t_out, y_out

def metodo_rk2(func, condicao_inicial, T, dt):
    """
    Resolve um PVI usando o método de Runge-Kutta de ordem 2.

    Parameters
    ----------
    func : callable
        Função f(t, x) do problema.
    condicao_inicial : list or tuple
        [t0, x0] condição inicial.
    T : float
        Tempo final.
    dt : float
        Passo de integração.

    Returns
    -------
    t_out : ndarray
        Vetor de tempos.
    y_out : ndarray
        Vetor de soluções aproximadas.
    """
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]

    n = int((T - t0) / dt)

    t_out = np.zeros(n + 1)
    y_out = np.zeros(n + 1)

    # Define o primeiro ponto da solução (a condição inicial)
    t_out[0] = t0
    y_out[0] = y0

    for i in range(n):
        # Pega os valores do passo anterior (usando o índice i)
        ti = t_out[i]
        yi = y_out[i]

        f1 = dt*func(ti, yi)
        f2 = dt*func(ti + dt, yi + f1)

        # Fórmula do método de Runge-Kutta de ordem 2
        y_out[i + 1] = yi + (1 / 2) * (f1 + f2)

        # Atualiza o vetor de tempo
        t_out[i + 1] = ti + dt

    return t_out, y_out

def metodo_rk3(func, condicao_inicial, T, dt):
    """
    Resolve um PVI usando o método de Runge-Kutta de ordem 3.

    Parameters
    ----------
    func : callable
        Função f(t, x) do problema.
    condicao_inicial : list or tuple
        [t0, x0] condição inicial.
    T : float
        Tempo final.
    dt : float
        Passo de integração.

    Returns
    -------
    t_out : ndarray
        Vetor de tempos.
    y_out : ndarray
        Vetor de soluções aproximadas.
    """
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]

    n = int((T - t0) / dt)

    t_out = np.zeros(n + 1)
    y_out = np.zeros(n + 1)

    # Define o primeiro ponto da solução (a condição inicial)
    t_out[0] = t0
    y_out[0] = y0

    A = np.array([
        [0,   0,   0],
        [0.5, 0,   0],
        [-1,  2,   0]
    ])
    B = np.array([1/6, 4/6, 1/6])
    C = np.array([0, 0.5, 1])

    for i in range(n):
        # Pega os valores do passo anterior (usando o índice i)
        ti = t_out[i]
        yi = y_out[i]

        f = np.zeros(len(C))
        for j in range(len(C)):
            tj = ti + C[j]*dt
            yj = yi + np.dot(A[j, :j], f[:j])
            f[j] = dt * func(tj, yj)

        y_out[i + 1] = yi + np.dot(B, f)

        # Atualiza o vetor de tempo
        t_out[i + 1] = ti + dt

    return t_out, y_out

def metodo_rk2_modificado(func, condicao_inicial, T, dt):
    """
    Resolve um PVI usando o método de Runge-Kutta de ordem 2 modificado.

    Parameters
    ----------
    func : callable
        Função f(t, x) do problema.
    condicao_inicial : list or tuple
        [t0, x0] condição inicial.
    T : float
        Tempo final.
    dt : float
        Passo de integração.

    Returns
    -------
    t_out : ndarray
        Vetor de tempos.
    y_out : ndarray
        Vetor de soluções aproximadas.
    """
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]
    alpha = .5
    beta = .5
    w2 = 1 / (2 * alpha)
    w1 = 1 - w2
    n = int((T - t0) / dt)

    t_out = np.zeros(n + 1)
    y_out = np.zeros(n + 1)

    # Define o primeiro ponto da solução (a condição inicial)
    t_out[0] = t0
    y_out[0] = y0

    for i in range(n):
        # Pega os valores do passo anterior (usando o índice i)
        ti = t_out[i]
        yi = y_out[i]

        # Cálculo dos k1 e k2
        f1 = dt*func(ti, yi)
        f2 = dt*func(ti + dt*alpha, yi + f1*beta)

        # Fórmula do método de Runge-Kutta de ordem 2 modificado
        y_out[i + 1] = yi + w1*f1 + w2*f2

        # Atualiza o vetor de tempo
        t_out[i + 1] = ti + dt

    return t_out, y_out

def metodo_rk4(func, condicao_inicial, T, dt):
    """
    Resolve um PVI usando o método de Runge-Kutta de ordem 4.

    Parameters
    ----------
    func : callable
        Função f(t, x) do problema.
    condicao_inicial : list or tuple
        [t0, x0] condição inicial.
    T : float
        Tempo final.
    dt : float
        Passo de integração.

    Returns
    -------
    t_out : ndarray
        Vetor de tempos.
    y_out : ndarray
        Vetor de soluções aproximadas.
    """
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]

    n = int((T - t0) / dt)
    
    t_out = np.zeros(n + 1)
    y_out = np.zeros(n + 1)

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
        f2 = dt * func(ti + C[1]*dt, yi + A[1,0]*f1)
        f3 = dt * func(ti + C[2]*dt, yi + A[2,1]*f2)
        f4 = dt * func(ti + C[3]*dt, yi + A[3,2]*f3)

        y_out[i + 1] = yi + B[0]*f1 + B[1]*f2 + B[2]*f3 + B[3]*f4
        t_out[i + 1] = ti + dt

    return t_out, y_out

def metodo_rk6(func, condicao_inicial, T, dt):
    """
    Resolve um PVI usando o método de Runge-Kutta de ordem 6.

    Parameters
    ----------
    func : callable
        Função f(t, x) do problema.
    condicao_inicial : list or tuple
        [t0, x0] condição inicial.
    T : float
        Tempo final.
    dt : float
        Passo de integração.

    Returns
    -------
    t_out : ndarray
        Vetor de tempos.
    y_out : ndarray
        Vetor de soluções aproximadas.
    """
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]

    n = int(np.ceil((T - t0) / dt))

    t_out = np.zeros(n + 1)
    y_out = np.zeros(n + 1)

    # Define o primeiro ponto da solução (a condição inicial)
    t_out[0] = t0
    y_out[0] = y0

    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/3, 0, 0, 0, 0, 0, 0],
        [0, 2/3, 0, 0, 0, 0, 0],
        [1/12, 1/3, -1/12, 0, 0, 0, 0],
        [-1/16, 9/8, -3/16, -3/8, 0, 0, 0],
        [0, 9/8, -3/8, -3/4, 1/2, 0, 0],
        [9/44, -9/11, 63/44, 18/11, 0, -16/11, 0]
    ])
    B = np.array([11/120, 0, 27/40, 27/40, -4/15, -4/15, 11/120])
    C = np.array([0.0, 1/3, 2/3, 1/3, 1/2, 1/2, 1])

    for i in range(n):
        # Pega os valores do passo anterior (usando o índice i)
        ti = t_out[i]
        yi = y_out[i]

        f = np.zeros(len(C))
        for j in range(len(C)):
            tj = ti + C[j]*dt
            yj = yi + np.dot(A[j, :j], f[:j])
            f[j] = dt * func(tj, yj)

        y_out[i + 1] = yi + np.dot(B, f)

        # Atualiza o vetor de tempo
        t_out[i + 1] = ti + dt

    return t_out, y_out

def metodo_rk(func, condicao_inicial, T, dt0, tol=1e-6, metodo='RKF45'):
    """
    Resolve um PVI usando métodos Runge-Kutta com passo adaptativo.

    Permite escolher entre os métodos de Fehlberg (RKF45) ou Dormand-Prince (DOPRI54).
    Utiliza as matrizes de coeficientes A, B4, B5 e C para calcular as etapas intermediárias
    e faz controle adaptativo do passo com base no erro estimado entre as soluções de ordem 4 e 5.

    Parameters
    ----------
    func : callable
        Função f(t, x) do problema.
    condicao_inicial : list or tuple
        [t0, x0] condição inicial.
    T : float
        Tempo final.
    dt0 : float
        Passo inicial de integração.
    tol : float, optional
        Tolerância para o controle de erro (default é 1e-6).
    metodo : str, optional
        'RKF45' para Fehlberg ou 'DOPRI54' para Dormand-Prince (default é 'RKF45').

    Returns
    -------
    t_out : ndarray
        Vetor de tempos.
    y_out : ndarray
        Vetor de soluções aproximadas (ordem 5).
    n : int
        Número de passos realizados.
    ite : int
        Número total de iterações internas de ajuste de passo.
    """
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]
    n = 0
    ite = 0

    t_out = [t0]
    y_out = [y0]

    t = t0
    y = y0

    match metodo:
        case 'RKF45':
            A = np.array([
                [0,           0,           0,           0,           0,          0],
                [1/4,         0,           0,           0,           0,          0],
                [3/32,        9/32,        0,           0,           0,          0],
                [1932/2197,  -7200/2197,   7296/2197,   0,           0,          0],
                [439/216,    -8,           3680/513,   -845/4104,    0,          0],
                [-8/27,       2,          -3544/2565,   1859/4104,  -11/40,      0]
            ])
            B4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
            B5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
            C = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])

        case 'DOPRI54':
            A = np.array([
                [0, 0, 0, 0, 0, 0, 0],
                [1/5, 0, 0, 0, 0, 0, 0],
                [3/40, 9/40, 0, 0, 0, 0, 0],
                [44/45, -56/15, 32/9, 0, 0, 0, 0],
                [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
                [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
                [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
            ])
            B5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
            B4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
            C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])


    while t < T:
        dt = dt0
        if t + dt > T:
            dt = T - t

        f = np.zeros(len(C))
        for i in range(len(C)):
            ti = t + C[i]*dt
            yi = y + np.dot(A[i, :i], f[:i])
            f[i] = dt * func(ti, yi)

        y4 = y + np.dot(B4,f)
        y5 = y + np.dot(B5, f)
        erro = np.abs(y5 - y4)

        # Controle adaptativo do passo
        i = 0
        while erro >= tol and i < 10:
            dt = .9 * dt * (tol / erro)**(1/5) # Kincaid e Cheney 
            if t + dt > T:
                dt = T - t

            for i in range(len(C)):
                ti = t + C[i]*dt
                yi = y + np.dot(A[i, :i], f[:i])
                f[i] = dt * func(ti, yi)

            y4 = y + np.dot(B4,f)
            y5 = y + np.dot(B5, f)
            erro = np.abs(y5 - y4)
            i += 1

        if i == 10 and erro >= tol:
            print("Aviso: número máximo de iterações internas atingido, erro ainda acima da tolerância!")

        t += dt
        y = y5
        t_out.append(t)
        y_out.append(y)
        n += 1
        ite += 1 + i

    return np.array(t_out), np.array(y_out), n, ite

def metodo_rk23(func, condicao_inicial, T, dt0, tol=1e-6):
    """
    Resolve um PVI usando o método de Bogacki-Shampine (RK23) com passo adaptativo.

    Parameters
    ----------
    func : callable
        Função f(t, x) do problema.
    condicao_inicial : list or tuple
        [t0, x0] condição inicial.
    T : float
        Tempo final.
    dt0 : float
        Passo inicial de integração.
    tol : float, optional
        Tolerância para o controle de erro (default é 1e-6).

    Returns
    -------
    t_out : ndarray
        Vetor de tempos.
    y_out : ndarray
        Vetor de soluções aproximadas (ordem 3).
    n : int
        Número de passos realizados.
    ite : int
        Número total de iterações internas de ajuste de passo.
    """
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]
    n = 0
    ite = 0

    t_out = [t0]
    y_out = [y0]

    t = t0
    y = y0

    # Coeficientes de Bogacki-Shampine (RK23)
    A = np.array([
        [0,   0,   0, 0],
        [0.5, 0,   0, 0],
        [0,   0.75, 0, 0],
        [2/9, 1/3, 4/9, 0]
    ])
    B3 = np.array([2/9, 1/3, 4/9, 0])
    B2 = np.array([7/24, 1/4, 1/3, 1/8])
    C = np.array([0, 0.5, 0.75, 1])

    while t < T:
        dt = dt0
        if t + dt > T:
            dt = T - t

        f = np.zeros(4)
        for i in range(4):
            ti = t + C[i]*dt
            yi = y + np.dot(A[i, :i], f[:i])
            f[i] = dt * func(ti, yi)

        y3 = y + np.dot(B3, f)
        y2 = y + np.dot(B2, f)
        erro = np.abs(y3 - y2)

        # Controle adaptativo do passo
        i = 0
        while erro >= tol and i < 10:
            dt = .9 * dt * (tol / erro)**(1/3 + 1) # Kincaid e Cheney
            if t + dt > T:
                dt = T - t

            for j in range(4):
                tj = t + C[j]*dt
                yj = y + np.dot(A[j, :j], f[:j])
                f[j] = dt * func(tj, yj)

            y3 = y + np.dot(B3, f)
            y2 = y + np.dot(B2, f)
            erro = np.abs(y3 - y2)
            i += 1

        if i == 10 and erro >= tol:
            print("Aviso: número máximo de iterações internas atingido, erro ainda acima da tolerância!")

        t += dt
        y = y3
        t_out.append(t)
        y_out.append(y)
        n += 1
        ite += 1 + i

    return np.array(t_out), np.array(y_out), n, ite

def metodo_ab2(func, condicao_inicial, T, dt):
    """
    Resolve um PVI usando o método de Adams-Bashforth de ordem 2.

    Parameters
    ----------
    func : callable
        Função f(t, x) do problema.
    condicao_inicial : list or tuple
        [t0, x0] condição inicial.
    T : float
        Tempo final.
    dt : float
        Passo de integração.

    Returns
    -------
    t_out : ndarray
        Vetor de tempos.
    y_out : ndarray
        Vetor de soluções aproximadas.
    """
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]

    n = int((T - t0) / dt)

    t_out = np.zeros(n + 1)
    y_out = np.zeros(n + 1)

    # Define o primeiro ponto da solução (a condição inicial)
    t_out[0] = t0
    y_out[0] = y0

    # Primeiro passo usando rk4
    t_out[1] = t_out[0] + dt
    y_out[1] = metodo_rk4(func, [t_out[0], y_out[0]], t_out[1], dt)[1][1]

    for i in range(1, n):
        ti = t_out[i]
        yi = y_out[i]
        t_ant = t_out[i - 1]
        y_ant = y_out[i - 1]

        # Fórmula do método de Adams-Bashforth de ordem 2
        y_out[i + 1] = yi + (dt / 2) * (3*func(ti, yi) - func(t_ant, y_ant))

        # Atualiza o vetor de tempo
        t_out[i + 1] = ti + dt

    return t_out, y_out

def metodo_ab3(func, condicao_inicial, T, dt):
    """
    Resolve um PVI usando o método de Adams-Bashforth de ordem 3.

    Parameters
    ----------
    func : callable
        Função f(t, x) do problema.
    condicao_inicial : list or tuple
        [t0, x0] condição inicial.
    T : float
        Tempo final.
    dt : float
        Passo de integração.

    Returns
    -------
    t_out : ndarray
        Vetor de tempos.
    y_out : ndarray
        Vetor de soluções aproximadas.
    """
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]

    n = int((T - t0) / dt)

    t_out = np.zeros(n + 1)
    y_out = np.zeros(n + 1)

    # Define o primeiro ponto da solução (a condição inicial)
    t_out[0] = t0
    y_out[0] = y0

    # Primeiro passo usando RK4
    t_out[1] = t_out[0] + dt
    y_out[1] = metodo_rk6(func, [t_out[0], y_out[0]], t_out[1], dt)[1][1]

    # Segundo passo usando RK4
    t_out[2] = t_out[1] + dt
    y_out[2] = metodo_rk6(func, [t_out[1], y_out[1]], t_out[2], dt)[1][1]

    for i in range(2, n):
        ti = t_out[i]
        yi = y_out[i]
        t_ant1 = t_out[i - 1]
        y_ant1 = y_out[i - 1]
        t_ant2 = t_out[i - 2]
        y_ant2 = y_out[i - 2]

        # Fórmula do método de Adams-Bashforth de ordem 3
        y_out[i + 1] = yi + (dt / 12) * (23*func(ti, yi) - 16*func(t_ant1, y_ant1) + 5*func(t_ant2, y_ant2))

        # Atualiza o vetor de tempo
        t_out[i + 1] = ti + dt

    return t_out, y_out

def metodo_ab4(func, condicao_inicial, T, dt):
    """
    Resolve um PVI usando o método de Adams-Bashforth de ordem 4.

    Parameters
    ----------
    func : callable
        Função f(t, x) do problema.
    condicao_inicial : list or tuple
        [t0, x0] condição inicial.
    T : float
        Tempo final.
    dt : float
        Passo de integração.

    Returns
    -------
    t_out : ndarray
        Vetor de tempos.
    y_out : ndarray
        Vetor de soluções aproximadas.
    """
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]

    n = int((T - t0) / dt)

    t_out = np.zeros(n + 1)
    y_out = np.zeros(n + 1)

    # Define o primeiro ponto da solução (a condição inicial)
    t_out[0] = t0
    y_out[0] = y0

    # Primeiro passo usando RK4
    t_out[1] = t_out[0] + dt
    y_out[1] = metodo_rk6(func, [t_out[0], y_out[0]], t_out[1], dt)[1][1]

    # Segundo passo usando RK4
    t_out[2] = t_out[1] + dt
    y_out[2] = metodo_rk6(func, [t_out[1], y_out[1]], t_out[2], dt)[1][1]

    # Terceiro passo usando RK4
    t_out[3] = t_out[2] + dt
    y_out[3] = metodo_rk6(func, [t_out[2], y_out[2]], t_out[3], dt)[1][1]

    for i in range(3, n):
        ti = t_out[i]
        yi = y_out[i]
        t_ant1 = t_out[i - 1]
        y_ant1 = y_out[i - 1]
        t_ant2 = t_out[i - 2]
        y_ant2 = y_out[i - 2]
        t_ant3 = t_out[i - 3]
        y_ant3 = y_out[i - 3]

        # Fórmula do método de Adams-Bashforth de ordem 4
        y_out[i + 1] = yi + (dt / 24) * (55*func(ti, yi) - 59*func(t_ant1, y_ant1) + 37*func(t_ant2, y_ant2) - 9*func(t_ant3, y_ant3))

        # Atualiza o vetor de tempo
        t_out[i + 1] = ti + dt

    return t_out, y_out

def metodo_ab6(func, condicao_inicial, T, dt):
    """
    Resolve um PVI usando o método de Adams-Bashforth de ordem 6.

    Parameters
    ----------
    func : callable
        Função f(t, x) do problema.
    condicao_inicial : list or tuple
        [t0, x0] condição inicial.
    T : float
        Tempo final.
    dt : float
        Passo de integração.

    Returns
    -------
    t_out : ndarray
        Vetor de tempos.
    y_out : ndarray
        Vetor de soluções aproximadas.
    """
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]

    n = int((T - t0) / dt)

    t_out = np.zeros(n + 1)
    y_out = np.zeros(n + 1)

    # Define o primeiro ponto da solução (a condição inicial)
    t_out[0] = t0
    y_out[0] = y0

    # Primeiro passo usando RK4
    t_out[1] = t_out[0] + dt
    y_out[1] = metodo_rk6(func, [t_out[0], y_out[0]], t_out[1], dt)[1][1]

    # Segundo passo usando RK4
    t_out[2] = t_out[1] + dt
    y_out[2] = metodo_rk6(func, [t_out[1], y_out[1]], t_out[2], dt)[1][1]

    # Terceiro passo usando RK4
    t_out[3] = t_out[2] + dt
    y_out[3] = metodo_rk6(func, [t_out[2], y_out[2]], t_out[3], dt)[1][1]

    # Quarto passo usando RK4
    t_out[4] = t_out[3] + dt
    y_out[4] = metodo_rk6(func, [t_out[3], y_out[3]], t_out[4], dt)[1][1]

    # Quinto passo usando RK4
    t_out[5] = t_out[4] + dt
    y_out[5] = metodo_rk6(func, [t_out[4], y_out[4]], t_out[5], dt)[1][1]

    for i in range(5, n):
        ti = t_out[i]
        yi = y_out[i]
        t_ant1 = t_out[i - 1]
        y_ant1 = y_out[i - 1]
        t_ant2 = t_out[i - 2]
        y_ant2 = y_out[i - 2]
        t_ant3 = t_out[i - 3]
        y_ant3 = y_out[i - 3]
        t_ant4 = t_out[i - 4]
        y_ant4 = y_out[i - 4]
        t_ant5 = t_out[i - 5]
        y_ant5 = y_out[i - 5]

        # Fórmula do método de Adams-Bashforth de ordem 6
        y_out[i + 1] = yi + (dt / 1440) * (4277*func(ti, yi) - 7923*func(t_ant1, y_ant1) + 9982*func(t_ant2, y_ant2) - 7298*func(t_ant3, y_ant3) + 2877*func(t_ant4, y_ant4) - 475*func(t_ant5, y_ant5))

        # Atualiza o vetor de tempo
        t_out[i + 1] = ti + dt

    return t_out, y_out

def metodo_am2(func, condicao_inicial, T, dt):
    """
    Resolve um PVI usando o método de Adams-Moulton de ordem 2.

    Parameters
    ----------
    func : callable
        Função f(t, x) do problema.
    condicao_inicial : list or tuple
        [t0, x0] condição inicial.
    T : float
        Tempo final.
    dt : float
        Passo de integração.

    Returns
    -------
    t_out : ndarray
        Vetor de tempos.
    y_out : ndarray
        Vetor de soluções aproximadas.
    """
    t0 = condicao_inicial[0]
    y0 = condicao_inicial[1]

    n = int((T - t0) / dt)

    t_out = np.zeros(n + 1)
    y_out = np.zeros(n + 1)

    # Define o primeiro ponto da solução (a condição inicial)
    t_out[0] = t0
    y_out[0] = y0

    for i in range(1, n):
        ti = t_out[i]
        yi = y_out[i]

        # Fórmula do método de Adams-Moulton de ordem 2
        y_pred = metodo_rk2_modificado(func, (ti, yi), ti + dt, dt)[1][1]
        y_out[i + 1] = yi + (dt / 2) * (func(ti, yi) + func(ti + dt, y_pred))

        # Atualiza o vetor de tempo
        t_out[i + 1] = ti + dt

    return t_out, y_out
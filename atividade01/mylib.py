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
    return np.abs(np.divide(y_exato - y_aproximado, y_exato, where=np.abs(y_exato) > 1e-6))

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

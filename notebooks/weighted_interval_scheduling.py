"""
Algoritmo de agendamento de intervalos ponderados (weighted interval scheduling)
Implementação em O(n.lg(n)) usando programação dinâmica

fontes:
https://courses.cs.washington.edu/courses/cse521/13wi/slides/06dp-sched.pdf
https://www.cs.princeton.edu/~wayne/kleinberg-tardos/pdf/06DynamicProgrammingI.pdf

autor:
Rafael Rocha
"""

import numpy as np

def bin_search(arr, target):
    """
    O array 'arr' deve estar ordenado.
    retorna o índice em 'arr' do último elemento menor ou igual a target.
    se  todos os elementos de 'arr' forem maiores que target, retorna -1
    """
    return _bin_search(arr,target,0,len(arr))

def _bin_search(arr, target, start, end):
    """chamada recursiva de busca binária no intervalo [start..end]"""
    # se há apenas um elemento
    if (end-start) == 0:
        # e este elemento é maior que o que objetivo
        if arr[start] > target:
            # retorna elemento anterior
            return start-1
        # caso contrario retorna atual
        return start
    # busca elemento no meio do intervalo
    p = start + (end-start)//2
    # se elemento do meio do intervalo é maior que alvo
    if arr[p] > target:
        # busca nos elementos do lado esquerdo do intervalo
        return _bin_search(arr, target, start, p)
    else:
        # caso contrario busca nos elementos no intervalo direito
        return _bin_search(arr, target, p+1, end)

def _compute_predecessor(start, end):
    """
    'start' e 'end' sao os arrays de tempos de inicio e término
    os intervalos devem estar ordenados pelo tempo de término 'end'
    
    retorna vetor p,
        onde p[i] é o índice do último intervalo que não sobrepõe 
        o intervalo i
    """
    p = [bin_search(end, s) for s in start]
    return p
    
def compute_schedule(s, f, v):
    """
    s: vetor de tempo de inicio dos intervalos
    f: vetor de tempo de término dos intervalos
    v: vetor de pesos de cada intervalo
    
    retorna o vetor de índices dos intervalos disjuntos que
        possuem a maior soma de valores
        
    """
    #ordena vetores por tempo de término
    sort_end = np.argsort(f)
    end_s = s[sort_end]
    end_f = f[sort_end]
    end_v = v[sort_end]
    #computa predecessores 
    p = _compute_predecessor(end_s, end_f)
    
    n = len(end_s)
    
    # opt[i] guarda valor do melhor cronograma até intervalo i
    opt = np.zeros(n+1)
    # tail[i] é 1 se intervalo i foi escolhido no subproblema
    #   considerando os intervalos [0..i],
    #   0 caso contrário.
    tail = np.zeros(n+1)
    
    for j in np.arange(1,n+1,1):
        choices = (opt[j-1], end_v[j-1] + opt[p[j-1]+1])
        opt[j] = np.max(choices)
        tail[j] = np.argmax(choices)
    
    schedule = []
    idx = n
    while idx > 0:
        # se adicionou intervalo idx no subproblema
        if tail[idx] == 1:
            # adiciona no cronograma
            schedule.append(idx-1)
            # vai para o ultimo intervalo que não sobrepõe o escolhido
            idx = p[idx-1]+1
        else:
            # caso contrário a solução do subproblema atual é a mesma do anterior
            idx -= 1
            
    # retorna índices para a ordem apresentada nos vetores de entrada
    sort_end = np.array(sort_end)
    schedule = sort_end[schedule]
    
    return schedule
        
        
        
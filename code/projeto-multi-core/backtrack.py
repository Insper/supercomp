# coding: utf8
import numpy as np
import sys

def escolhe_alunos(prefs, aluno_projeto, vagas, satisfacao_atual=0, melhor=None, i=0):
    if i == len(aluno_projeto): # todos alunos tem projeto
        if melhor is None:
            print('Melhor:', aluno_projeto, satisfacao_atual, file=sys.stderr)
            melhor = aluno_projeto.copy(), satisfacao_atual
        if satisfacao_atual > melhor[1]:
            melhor = aluno_projeto.copy(), satisfacao_atual
            print('Melhor:', melhor, file=sys.stderr)
        return melhor

    for proj_atual in range(prefs.shape[1]):
        if vagas[proj_atual] > 0: # projeto prefs[j] tem vaga!
            vagas[proj_atual] -= 1
            aluno_projeto[i] = proj_atual

            melhor = escolhe_alunos(prefs, aluno_projeto, vagas, satisfacao_atual + prefs[i, proj_atual], melhor, i+1)

            aluno_projeto[i] = -1
            vagas[proj_atual] += 1

    return melhor

if __name__ == '__main__':
    n_alunos, n_projetos, n_choices = [int(p) for p in input().split()]
    prefs = np.zeros((n_alunos, n_projetos), np.uint32)
    for i in range(n_alunos):
        projs = [int(c) for c in input().split(' ')]
        for j, p in enumerate(projs):
            prefs[i, p] = pow(n_choices - j, 2)
    
    vagas = np.ones(n_projetos, np.uint) * 3 # 3 vagas por projeto
    aluno_projeto = np.ones(n_alunos, np.uint) * -1 # não escolheu projeto ainda

    melhor = escolhe_alunos(prefs, aluno_projeto, vagas)

    print(melhor[1], '1')
    print(' '.join([str(int(m)) for m in melhor[0]]))
    

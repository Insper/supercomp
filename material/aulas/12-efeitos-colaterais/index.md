# 11 - Scheduling e  Efeitos Colaterais

# Scheduling

Vamos começar compreendendo melhor os `schedulers` que existem no openmp.
No github, obtenha o arquivo `omp_schedulers.cpp`, compile e execute-o. Você deverá obter um output similar ao abaixo. Verifique na documentação do OpenMP ([link](https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/use-the-openmp-libraries.html)) e busque compreender melhor como funciona cada scheduler. 

```C++

default:               
****************                                                
                ****************                                
                                ****************                
                                                ****************

schedule(static):      
****************                                                
                ****************                                
                                ****************                
                                                ****************

schedule(static, 4):   
****            ****            ****            ****            
    ****            ****            ****            ****        
        ****            ****            ****            ****    
            ****            ****            ****            ****

schedule(static, 8):   
********                        ********                        
        ********                        ********                
                ********                        ********        
                        ********                        ********

schedule(dynamic):     
**  ******** ** ***** * ** ****** * *  ****  *   *  **  *  *  * 
   *        *        *             *  *         * **   *        
  *            *       *         *   *     ** **      *  *  ** *
                          *                               *     

schedule(dynamic, 1):  
* **  ** * **  *   *  * * * ** * *    *** *  * * *******  * *  *
 *   *       *         * *        ** *     *            **   *  
    *   * *   * *** **     *  * *   *    *  * * *          *  * 
                                                                

schedule(dynamic, 4):  
****    ********************************    ********************
    ****                                                        
                                                                
                                        ****                    

schedule(dynamic, 8):  
********                ****************************************
        ********                                                
                ********                                        
                                                                

schedule(guided):      
****************            ****************     ***************
                ************                                    
                                                                
                                            *****               

schedule(guided, 2):   
****************                     ***************************
                ************                                    
                            *********                           
                                                                

schedule(guided, 4):   
****************            ************************************
                                                                
                ************                                    
                                                                

schedule(guided, 8):   
                ************         ************************   
                            *********                           
****************                                             ***
                                                                

schedule(auto):        
****************                                                
                ****************                                
                                ****************                
                                                ****************

schedule(runtime):     
****************                                                
                ****************                                
                                ****************                
                                                ****************

```

# Revisitando `Parallel for`, `tasks` e `sections`

Agora que já conseguimos resolver problemas simples usando três abordagens diferentes, vamos aumentar a complexidade dos problemas tratados. Vimos três abordagens

* `parallel for` - útil para quando precisamos executar a mesma operação em um conjunto grande de dados.
* `tasks` - útil para paralelizar tarefas heterogêneas.
* `sections` - permite paralelizar tarefas heterogênas, com o controle adicional que uma `section` é executada apenas por uma thread.

Teremos então dois desafios relacionados a paralelizar programas que não são obviamente paralelizáveis.

## Cálculo do `pi` recursivo

Vamos iniciar com um código recursivo para cálculo do pi.

!!! example
    Examine o código em *pi_recursivo.cpp*. Procure entender bem o que está acontecendo antes de prosseguir.

!!! question short
    Onde estão as oportunidades de paralelismo? O código tem dependências?

!!! question medium
    Se o código tiver dependências, é possível refatorá-lo para eliminá-las?

!!! question medium
    Quantas níveis de chamadas recursivas são feitas? Quando o programa para de chamar recursivamente e faz sequencial?

Vamos agora tentar paralelizar o programa usando as duas técnicas.

### Usando `for` paralelo

!!! question short
    Em quais linhas pode haver oportunidade para usar `parallel for`?

!!! example
    Crie uma implementação do *pi_recursivo* usando for paralelo. Meça seu tempo e anote.

!!! example
    O número `MIN_BLK` afeta seu algoritmo? É melhor aumentá-lo ou diminuí-lo? 

!!! question short
    Os ganhos de desempenho foram significativos?

!!! question short
    Como você fez o paralelismo? Precisou definir o número do `for` manualmente ou conseguiu realizar a divisão automaticamente? Comente abaixo sua implementação.


### Usando `task`

Agora vamos usar `task`. Neste caso é vamos adotar a seguinte estratégia: usaremos tarefas para paralelizar as chamadas recursivas feitas em *pi_recursivo.cpp*. 

!!! example
    Crie uma implementação do *pi_recursivo* usando tarefas. Meça seu tempo e anote.
    
    **Dica**: se você precisar esperar tarefas pode usar a diretiva `#pragma omp taskwait`. Ela espera por todas as tarefas criadas pela thread atual.

!!! question short
    Os ganhos de desempenho foram significativos?

!!! question short
    Quantas tarefas foram criadas? Você escolheu essa valor como?

!!! example
    Tente números diferentes de tarefas e verifique se o desempenho melhora ou piora. Anote suas conclusões abaixo. 

### Comparação

!!! question short
    Compare seus resultados das duas abordagens. Anote abaixo seus resultados.

!!! warning
    É possível conseguir tempos muito parecidos com ambas, então se uma delas ficou muito mais lenta é hora de rever o que foi feito.


# Efeitos Colaterais 

Agora que já conseguimos resolver um problema um pouco mais complexo usando abordagens diferentes, vamos aumentar um pouco mais a complexidade dos problemas tratados.

No código `pi_recursivo.cpp` tínhamos uma variável global que podia ser eliminada do código mudando a função recursiva. Isso, porém, nem sempre é possível e precisamos lidar com estas situações.

## Um primeiro teste

Vamos iniciar trabalhando com o seguinte trecho de código (arquivo `vetor_insert.cpp`):

```cpp
std::vector<double> vec;
for (int i = 0; i < N; i++) {
	vec.push_back(conta_complexa(i));
}
```

Vamos supor agora que usaremos o seguinte comando para paralelizar o código acima usando OpenMP:

```
#pragma omp parallel for
```

!!! question choice
	A variável `i` é

	- [ ] shared
	- [x] private
	- [ ] firstprivate

!!! question choice
	A variável `vec` é

	- [x] shared
	- [ ] private
	- [ ] firstprivate

!!! question choice
	O código paralelizado rodaria sem dar erros? Os resultados seriam os esperados?

	- [ ] Sim, o `vector` é capaz de gerenciar os acessos simultâneos
	- [ ] O código acima roda sem erros, mas o conteúdo do vetor pode não estar correto ao fim do programa
	- [x] Não, o código acima dá erro ao executar.

	!!! details "Resposta"
		Rode e veja o que acontece ;)

!!! progress
	Clique após rodar o programa

Agora que vimos o que acontece, vamos consertar isso!

!!! danger
	Nosso código dá erro pois a operação `push_back` **modifica o vetor**!


Vamos ver então duas abordagens importantes para contornar esse problema.

## Seções críticas

Antes de começar, vamos aprender mais um aspecto de OpenMP: diretivas para compartilhamento de dados. Já vimos as 3 principais opções:

- `shared` - compartilhado entre threads
- `private` - privados entre threads
- `firstprivate` - Especifica que cada thread deve ter sua própria instância de uma variável e que a variável deve ser inicializada com o valor da variável antes da seção paralela.

 Podemos forçar a especificação de diretivas de compartilhamento para **todas** as variáveis usadas nas construções `omp parallel`, `omp task` e `omp parallel for`.

!!! tip
	Ao adicionarmos `default(none)` logo após as diretivas acima precisaremos especificar, para cada variável usada, sua diretiva de compartilhamento. Isso torna muito mais fácil identificar casos de compartilhamento indevido de dados.

	A partir desse ponto estaremos supondo que todo código criado usará `default(none)`.

A primeira abordagem usada terá a missão de indicar que um conjunto de linhas contém uma operação que possui efeitos colaterais. Dessa maneira, podemos evitar conflitos se **só permitirmos que essa região rode em uma thread por vez**.

Fazemos isso usando a diretiva `omp critical`:

```cpp
#pragma omp critical
{
	// código aqui dentro roda somente em uma thread por vez.
}
```

Se duas threads chegam ao mesmo tempo no bloco `critical`, uma delas ficará esperando até a outra acabar o bloco. Quando isso ocorrer a thread que esperou poderá prosseguir. Vamos tentar aplicar isso ao código de `vetor_insert.cpp`.

!!! example
	Use `omp critical` para solucionar os problemas de concorrência do código acima.

!!! question short
	Escreva abaixo o tempo que seu código levou para rodar.

!!! progress
	Clique após rodar seu código

Se sua implementação se parecer com o código abaixo, então é bem provável que a versão paralela na verdade tenha demorado o mesmo tempo ou mais que o original.

```cpp
....
#pragma omp parallel for default(none) shared(vec)
for (int i = 0; i < N; i++) {
	#pragma omp critical
	{
		vec.push_back(conta_complexa(i));
	}
}
....
```

!!! question short
	Analise o código novamente e tente explicar por que o programa não ganhou velocidade.

	!!! details "Resposta"
		A operação que produz efeitos colaterais é `vec.push_back`, mas nossa seção crítica envolve também a chamada `conta_complexa(i)`.

!!! example
	Modifique seu código de acordo com a resposta acima. Meça o desempenho e veja que agora há melhora.

Vamos analisar agora a ordem dos dados em `vec`.

!!! question short
	A ordem se mantém igual ao programa sequencial? Você consegue explicar por que?

	!!! details "Resposta"
		Não se mantém. Cada thread chega ao `push_back` em um momento diferente, logo a ordem em que os dados são adicionados no vetor muda.

## Manejo de conflitos usando pré-alocação de memória

Seções críticas são muito úteis quando não conseguimos evitar o compartilhamento de dados. Porém, elas são caras e e feitas especialmente para situações em que região crítica é pequena e chamada um número relativamente pequeno de vezes. 

!!! danger "Como regra, desejamos entrar na região crítica o menor número possível de vezes."

!!! question short
	Reveja o código do inicial da seção de efeitos colaterais. Seria possível reescrevê-lo para não usar `push_back`?

	!!! details "Resposta"
		Sim, bastaria alocar o vetor com tamanho `N` ao criá-lo. Assim poderíamos atribuir `conta_complexa` direto para a posição de memória desejada.

A estratégia acima é muito importante em alto desempenho e representa uma maneira de evitar seções críticas e sincronização.

!!! done "É sempre melhor alocar memória em blocos grandes antes do paralelismo do que alocar memória frequentemente dentro de regiões paralelas."

Note que fizemos isso na parte de tarefas: ao criarmos variáveis para cada tarefa preencher evitamos a necessidade de usar sincronização.

!!! example
	Modifique o programa para usar a ideia da questão anterior. Meça o desempenho e verifique que tudo funciona normalmente e mais rápido que o original.


# Dica

A Microsoft possui uma página bem completa sobre o `OpenMP`. Vale a pena conferir neste [link](https://docs.microsoft.com/pt-br/cpp/parallel/openmp/reference/openmp-library-reference?view=msvc-160).
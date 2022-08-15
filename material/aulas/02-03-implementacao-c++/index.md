# 02/03 - Implementação em C++

!!! pdf
    ![](slides.pdf)

A disciplina utilizará a linguagem C++ para implementação dos programas. Ela é muito usada em implementações de alto desempenho e possui recursos muito úteis e que simplificam a programação se comparada com C puro. Nas aulas 02 e 03 aprenderemos alguns desses recursos e os utilizaremos para implementação de algoritmos simples. 

!!! failure "Gabaritos e respostas"
    Este curso não fornece código de resposta para os exercícios de sala. Cada exercício é acompanhado de um algoritmo em pseudo-código e alguns pares de arquivos entrada/saída. Isto já é suficiente para que vocês verifiquem se sua solução está correta. 

    Boas práticas de programação serão demonstradas em exercícios corrigidos pelo professor durante o semestre.

## Compilação 

Programas em C++ são compilados com o comando `g++`. Ele funciona igual ao `gcc` que vocês já usaram em Desafios e Sistemas Hardware-Software.

```
$> g++ -Wall -O3 arquivo.cpp -o executavel
```

## Entrada e saída em C++

Em C usamos as funções `printf` para mostrar dados no terminal e `scanf` para ler dados. Em C++ essas funções também podem ser usadas, mas em geral são substituídas pelos objetos `std::cin` e `std::cout` (disponíveis no cabeçalho iostream). 

A maior vantagem de usar `cin` e `cout` é que não precisamos mais daquelas strings de formatação estranhas com `%d`, `%s` e afins. Podemos passar variáveis diretamente para a saída do terminal usando o operador `<<`. Veja um exemplo abaixo. 

```cpp
int a = 10;
double b = 3.2;
std::cout << "Saída: " << a << ";" << b << "\n";
```


!!! hint "E esse `std::`?"
    Em `C++` podemos ter várias funções, variáveis e objetos em geral com o mesmo nome. Para evitar que eles colidam e não se saiba a qual estamos nos referindo cada nome deve ser definido um `namespace` (literalmente *espaco de nomes*). Podemos ter `namespace`s aninhados.Por exemplo, `std::chrono` contém as funções relacionadas contagem de tempo durante a execução de um programa. 

    Todas as funções, classes e globais na biblioteca padrão estão definidas no espaço `std`. Se quisermos, podemos omitir escrever `std::` toda vez digitando `using namespace std`. Isso pode ser feito também com namespaces aninhados. 

A implementação de algoritmos definidos usando expressões matemáticas é uma habilidade importante neste curso.

!!! exercise
    Escreva um programa que receba um inteiro `n` e calcule a seguinte série.

    $$
    S = \sum_{i=0}^n \frac{1}{2^i}
    $$

    Mostre as primeiras 15 casas decimais de `S`. Veja a documentação de [`std::setprecision` aqui](http://cplusplus.com/reference/iomanip/setprecision/). 

    !!! answer
        Essa série converge para o número 2, mas sua resposta deverá ser sempre menor que este número. Logo, quanto maior `n` mais próxima sua resposta será. Seu programa deverá implementar algo como o algoritmo abaixo.

        ```
        leia inteiro n
        s = 0.0
        para i=0 até n
            s += 1 / (2 elevado a i)
        
        print(s)
        ```

## Vetores em C++ com Vector

A estrutura `std::vector` é um vetor dinâmico que tem funcionalidades parecidas com a lista de Python ou o `ArrayList` de Java. O código abaixo exemplifica seu uso e mostra algumas de suas funções. Note que omitimos o uso de `std` no código abaixo.

```cpp
int n;
cin >> n;
vector<double> vec;
for (int i = 0; i < n; i++) {
    vec.push_back(i * i)
}
cout << "Tamanho do vetor: " << vec.size() << "\n";
cout << "Primeiro elemento: " << vec.front() << "\n";
cout << "Último elemento: " << vec.back() << "\n";
cout << "Elemento 3: " << vec[2] << "\n";
```

Alguns pontos interessantes deste exemplo:

1. Não sabemos o tamanho de `vec` ao criá-lo. O método `push_back` aumenta ele quando necessário e não precisamos nos preocupar com isso. 
2. O número de elementos colocados no vetor é retornado pelo método `size()`
3. O acesso é feito exatamente igual ao array de C, usando os colchetes `[]`

!!! tip "E esse `<double>` na declaração?" 
    Em C++ tipos passados entre `< >` são usados para parametrizar tipos genéricos. Ou seja, um vetor pode guardar qualquer tipo de dado e precisamos indicar qual ao criá-lo. 

    Note que, portanto, um vetor `vector<int>` e um vetor `vector<double>` são considerados de tipos diferentes e não posso passar o primeiro para uma função esperando o segundo. 

!!! exercise 
    Crie um programa que lê um número inteiro `n` e depois lê `n` números fracionários $x_i$. Faça os seguintes cálculos e motre-os no terminal com 10 casas decimais. 

    $$\mu = \frac{1}{n} \sum_{i=1}^n x_i$$


    $$\sigma^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2$$

    !!! answer
         Use o programa `t4.py` para gerar entradas e saídas de teste para seu programa. 


!!! question short
    Você reconhece as fórmulas acima? Elas calculam quais medidas estatísticas?

    !!! answer
        Média e variância.

## Matrizes (versão 1)

Dados `N` pontos com coordenadas $(x_i, y_i)_{i=0}^N$, computar a matriz de distâncias $D$ tal que 

$$
D_{i,j} = \textrm{Distância entre } (x_i, y_i) \textrm{ e } (x_j, y_j)
$$

!!! tip
    Use `t6.py` para gerar os arquivos de entrada/saída da tarefa abaixo. 

!!! exercise
    Implemente um programa que calcule a matriz `D` acima. Sua entrada deverá estar no formato dos arquivos `t6-in-*.txt` e sua saída no formato dos arquivos `t6-out-*.txt`. Mostre as distâncias com 2 casas decimais.  

    **Dicas**:
    
    1. a maneira mais fácil (não necessariamente a melhor) de alocar uma matriz é usando um vetor em que cada elemento é outro vetor. 
    2. faça uma implementação o mais simples possível. Vamos melhorá-la nas próximas tarefas.

    !!! answer
        ```
        leia inteiro N
        leia vetores X e Y 

        seja D uma matriz NxN

        para i=1..N:
            para j=1..N:
                DX = X[i] - X[j]
                DY = Y[i] - Y[j]
                D[i,j] = sqrt(DX*DX + DY*DY)
        ```


!!! question medium
    Anote abaixo o tempo de execução para os arquivos `t6-in-*.txt` e `t6-out-*.txt`

!!! question short
    Qual é a complexidade computacional de sua implementação? 

## Referências e passagem de dados

Na parte anterior fizemos nosso programa inteiro no `main`. Vamos agora organizá-lo melhor. 

!!! exercise
    Crie uma função `calcula_distancias` que recebe a matriz e os dados recebidos na entrada e a preenche. Sua função não deverá retornar nenhum valor. 

    Ao terminar, meça o tempo de execução para o arquivo `t6-out-4.txt`.

    !!! answer
        Aqui podem ocorrer dois problemas:

        1. Seu programa deu "Segmentation Fault". 
        2. Seu programa rodou até o fim, mas a saída é vazia (ou cheia de 0).

        O problema em si depende de como você fez o `for` duplo para mostrar os resultados. De qualquer maneira, simplesmente mover código para uma outra função não funciona neste caso. 

Ambos problemas descritos na solução são previsíveis e ocorrem pela mesma razão: **ao passar um `vector` para uma função é feita uma cópia de seu conteúdo**. Ou seja, a matriz usada dentro de `calcula_distancias` não é a mesma do `main`! 

Isto é considerado uma *feature* em `C++`: por padrão toda variável é passada **por cópia**. Isto evita que uma função modifique um valor sem que o código chamador fique sabendo. 

Em *C* podemos passar variáveis **por referência** passando um ponteiro para elas. Apesar de funcional, isso não é muito prático pois temos que acessar a variável sempre usando `*`.  Em *C++* temos um novo recurso: referências. Ao declarar uma variável como uma referência crio uma espécie de *ponteiro constante* que sempre acessa a variável apontada. Veja o exemplo abaixo.

```cpp
int x = 10;
int &ref = x; // referências são declaradas colocando & na frente do nome da variável
// a partir daqui ref e x representam a mesma variável
ref = 15;
cout << x << "\n"; // 15
```

O mesmo poderia ser feito com ponteiros (como mostrado abaixo). A grande vantagem da referência é que não precisamos usar `*ref` para nos referirmos à variável `x`! Na atribuição também podemos usar direto `int &ref = x`, o que torna o código mais limpo e fácil de entender.  

```cpp
int x = 10;
int *ref = &x; // precisamos de &x para apontar ref para a variável x
*ref = 15; // precisamos indicar *ref para atribuir a variável x
cout << x << "\n"; // 15
```

!!! tip "Dicas"
    Note que uma referência tem que ser inicializada com a variável a que ela se refere. Ou seja, ao declarar tenho que já indicar a variável destino e esse destino não pode ser modificado. 

!!! exercise
    Modifique sua função para usar referências. Verifique que ele volta a funcionar e que seu tempo de execução continua parecido com a versão que rodava no `main`.

    !!! answer
        Basta adicionar `&` na frente dos nomes dos argumentos (vetores x, y e matriz). A chamada da função não muda. 

!!! tip "Dica"
    Em *C++* precisamos estar sempre atentos à maneira que passamos os dados. Se não indicarmos será por cópia. Para compartilhar o mesmo objeto entre várias funções usamos referências `&`. 


## Uma primeira otimização

Nossa primeira implementação é bastante direta da definição e não tenta ser eficiente. 

!!! question short
    Analisando a definiçao da Tarefa 1, como seria possível economizar trabalho?

    !!! answer
        Podemos ver que a matriz `D` é simétrica. Ou seja, `D[i,j] == D[j,i]`. Isso significa que poderíamos calcular só um deles e copiar o valor para a outra posição.

!!! question short
    Como isso poderia ser usado para melhorar o tempo de execução de `calcula_distancias`?

!!! question short
    Seu programa criado na tarefa 1 consegue ser adaptado para implementar sua ideia da questão 
    anterior? O que precisaria ser modificado?

    !!! answer
        Duas respostas são possíveis e corretas aqui:

        1. Preciso checar se o `i > j` e usar o valor já calculado de `D[j,i]`.
        
        2. É preciso alocar a matriz inteira antes de começar. Se formos dando `push_back` linha a linha não conseguimos atribuir um valor ao mesmo tempo a `D[i,j]` e `D[j,i]`, já que um deles ainda não terá sido criado. 

Baseado na resposta acima vamos tentar nossa primeira otimização: só vamos calcular `D[i,j]` para `i <= j` (ou seja, só a metade "de cima" de `D`).

!!! exercise
    Use a estratégia acima para evitar calcular a matriz inteira. Verifique se houve melhora no tempo do teste `t6-in-3.txt`.

    **Dica**: tente de novo usar a ideia mais simples possível e implemente adicionando um so `if` no seu programa.

    !!! answer
        Não deverá haver ganho de desempenho significativo. Veremos exatamente o por que na próxima aula. 

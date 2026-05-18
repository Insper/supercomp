# Transferência de Arquivos entre sua máquina e o Cluster Franky

O comando `scp` (Secure Copy Protocol) é uma ferramenta segura e eficiente para transferir arquivos entre sua máquina local e um servidor remoto, como o cluster Franky. Ele utiliza o protocolo SSH para realizar a transferência de arquivos, garantindo a segurança dos dados durante o processo.

## Pré-requisitos

Antes de começar, certifique-se de que:
- Você tenha acesso ao cluster Franky via SSH.
- O comando `scp` esteja instalado em sua máquina (a maioria dos sistemas operacionais baseados em Unix/Linux já possuem o `scp` por padrão).

## Transferindo um Arquivo da Sua Máquina para o Cluster Franky

Para transferir um arquivo do seu computador para o cluster Franky, você pode usar o seguinte comando:

```bash
scp /caminho/local/do/arquivo.txt usuario@ip_do_cluster:/caminho/remoto/destino/
```
Este comando copia o arquivo `meu_arquivo.txt` do diretório local `/home/user/` para o diretório `/home/usuario/destino/` no cluster Franky.

### Explicação dos Parâmetros:
- `/caminho/local/do/arquivo.txt`: Caminho completo do arquivo na sua máquina local que você deseja transferir.
- `usuario@ip_do_cluster`: Seu nome de usuário e o endereço de ip do cluster Franky.
- `/caminho/remoto/destino/`: O diretório de destino no cluster Franky onde você deseja salvar o arquivo.


## Transferindo um Arquivo do Cluster Franky para sua Máquina

Para copiar um arquivo do cluster Franky para a sua máquina local, use o seguinte comando no seu terminal:

```bash
scp usuario@ip_do_cluster:/caminho/remoto/do/arquivo.txt /caminho/local/destino/
```
Este comando copia o arquivo `arquivo_remoto.txt` do diretório `/home/usuario/` no cluster Franky para o diretório `/home/user/destino_local/` na sua máquina.

### Explicação dos Parâmetros:
- `usuario@franky.cluster:/caminho/remoto/do/arquivo.txt`: O caminho completo do arquivo no cluster Franky que você deseja transferir para sua máquina local.
- `/caminho/local/destino/`: O diretório de destino na sua máquina local onde você deseja salvar o arquivo.


- **Transferência Recursiva:** Para transferir diretórios inteiros (incluindo subdiretórios e arquivos), use a opção `-r`:

```bash
scp -r /caminho/local/do/diretorio/ usuario@ip_do_cluster:/caminho/remoto/destino/
```

O comando `scp` é uma ferramenta poderosa para transferir arquivos entre sua máquina e o cluster Franky de forma segura e eficiente. Com as instruções e exemplos fornecidos neste guia, você deve ser capaz de realizar transferências de arquivos com facilidade.


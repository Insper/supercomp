Para utilizar as chaves ssh elas devem estas com as permissões de acesso no modo
correto. 

# No Linux e Mac: 

```sh
sudo chmod 400 nome_da_chave
```

# No Windows: 

```ps1
# Source - https://stackoverflow.com/a/43317244
icacls.exe nome_da_chave /reset
icacls.exe nome_da_chave /GRANT:R "$($env:USERNAME):(R)"
icacls.exe nome_da_chave /inheritance:r
```


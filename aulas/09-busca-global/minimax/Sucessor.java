


/*
 * CLASSE SUCESSOR  - GERA OS ESTADOS DO JOGO DA VELHA 
 */

public class Sucessor
{
  int[][] tabuleiro;
  int utilidade;
  
  /*
   *  Metodo Construtor
   */
  public Sucessor (int[][] tab)
  {
    /*
     * Cria um novo tabuleiro, baseado no que foi passado
     */
    int tam = tab.length;
    tabuleiro = new int[tam][tam];

    for (int i = 0; i < tam; i++)
      for (int j = 0; j < tam; j++)
        tabuleiro[i][j] = tab[i][j];
  }
}

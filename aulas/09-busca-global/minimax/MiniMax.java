


/*
 * CLASSE MINIMAX - ALGORITMO DE BUSCA COMPETITIVA
 */



import java.util.ArrayList;
import java.util.Collections;

public class MiniMax
{
  /*
   * Lista de Sucessores. Esta lista é armazenada utilizando 
   * um ArrayList
   */
  static ArrayList<Sucessor> sucessores = new ArrayList<Sucessor> ();
  int tam, maxProf;
  
  /*
   * Construtor recebe o tamanho do tabuleiro e a profundidade máxima da busca
   */
  public MiniMax (int tam, int maxProf)
  {
    this.tam = tam;
    if (maxProf > 0)
      this.maxProf = maxProf;
    else
      this.maxProf = Integer.MAX_VALUE; //Recebe o maior valor de um inteiro. 
  }
  
  /*
   * Metodo de decisao do MiniMax
   */
  public int[][] decisao_minimax (int[][] tab)
  {
    /*
     * Limpa os sucessores
     */
    sucessores.clear ();

    /*
     * Recebe a utilidade máxima
     */
    int v = valor_max (tab, true, 1);
    
    /*
     * Percorre a lista em busca do primeiro sucessor com utilidade máxima
     */
    for (Sucessor s: sucessores)
      if (s.utilidade == v)
        return s.tabuleiro;
    
    return tab;
  }
  
  public int valor_max (int[][] tab, boolean prim, int prof)
  {
    /*
     * Se a profundidade for maior que a máxima ou o jogo acabou, retorna a
     * utilidade
     */
    if (prof++ > maxProf || teste_terminal (tab))
      return utilidade (tab);
    
    /*
     * Atribui o menor valor de um inteiro para v ( - infinito)
     */
    int v = Integer.MIN_VALUE;
    
    /*
     * Percorre os n�s sucessores de MAX
     */
    for (Sucessor s: gerar_sucessores (tab, 1))
    {
      v = Math.max(v, valor_min (s.tabuleiro, prof));
      s.utilidade = v;
      /*
       * Se forem os primeiros sucessores, adiciona na lista de sucessores...
       */
      if (prim)
        sucessores.add(s);
    }
    
    return v;
  }
  
  public int valor_min (int[][] tab, int prof)
  {
    /*
     * Se a profundidade for maior que a m�xima ou o jogo acabou, retorna a
     * utilidade
     */
    if (prof++ > maxProf || teste_terminal (tab))
      return utilidade (tab);
    
    /*
     * Atribui +Infinito
     */
    int v = Integer.MAX_VALUE;
    
    /*
     * Percorre os n�s sucessores de MIN
     */
    for (Sucessor s: gerar_sucessores (tab, -1))
    {
      v = Math.min(v, valor_max (s.tabuleiro, false, prof));
      s.utilidade = v;
    }
    
    return v;
  }

  /*
   * Gera os sucessores de um jogador, a partir do estado atual
   */
  public ArrayList<Sucessor> gerar_sucessores (int[][] tab, int v)
  {
    ArrayList<Sucessor> suc = new ArrayList<Sucessor> ();
    for (int i = 0; i < tam; i++)
    {
      for (int j = 0; j < tam; j++)
      {        
        if (tab[i][j] == 0)
        {
          tab[i][j] = v;
          suc.add(new Sucessor (tab));
          tab[i][j] = 0;
        }
      }
    }
    
    return suc;
  }
  
  /*
   * Verifica se chegou em algum estado terminal e caso afirmativo finaliza o jogo
   */
  public boolean teste_terminal (int[][] tab)
  {
    return (ganhou (tab, 1) || ganhou (tab, -1) || semEspaco (tab));
  }
  
  /*
   * Retorna a utilidade
   */
  public int utilidade (int[][] tab)
  {
    if (ganhou (tab, 1))
      return 1;
    else if (ganhou (tab, -1))
      return -1;
    else
      return 0;
  }
  
  /*
   * Verifica se jogador ganhou
   */
  public boolean ganhou (int[][] tab, int v)
  {
    for (int i = 0; i < tam; i++)
      if (ganhouLinha (tab, i, v) || ganhouColuna (tab, i, v))
        return true;
    
    if (ganhouDiag1(tab, v) || ganhouDiag2 (tab, v))
      return true;
    
    return false;
  }
  
  /*
   * Ganhou na sequencia de linhas?
   */
  private boolean ganhouLinha (int[][] tab, int l, int v)
  {
    for (int i = 0; i < tam; i++)
      if (tab[l][i] != v)
        return false;
    
    return true;
  }
  
  /*
   * Ganhou na sequencia de colunas?
   */
  private boolean ganhouColuna (int[][] tab, int c, int v)
  {
    for (int i = 0; i < tam; i++)
      if (tab[i][c] != v)
        return false;
    
    return true;
  }
  
  /*
   * Ganhou na sequencia diagonal principal?
   */
  private boolean ganhouDiag1 (int[][] tab, int v)
  {
    for (int i = 0; i < tam; i++)
      if (tab[i][i] != v)
        return false;
    
    return true;
  }
  
  /*
   * Ganhou na sequencia diagonal secundaria?
   */
  private boolean ganhouDiag2 (int[][] tab, int v)
  {
    for (int i = 0; i < tam; i++)
      if (tab[(tam-1)-i][i] != v)
        return false;
    
    return true;
  }
  
  /*
   * Nao tem mais espacos restantes no tabuleiro..
   */
  public boolean semEspaco (int[][] tab)
  {    
    for (int l = 0; l < tam; l++)
      for (int c = 0; c < tam; c++)
        if (tab[l][c] == 0)
          return false;
    
    return true;
  }
}

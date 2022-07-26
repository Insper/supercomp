

/*
 * CLASSE VELHA  - CLASSE PRINCIPAL DA APLICACAO 
 */


//Biblioteca Scanner para captura da jogada do usu�rio
import java.util.Scanner;

public class Velha
{  
  /*
   * CONSTANTES UTILIDAZAS
   * TAM -> Tamanho do Tabuleiro
   * PROF -> Profundidade m�xima da busca no MiniMax. Se PROF = -1 o algoritmo
   * minimax ir� buscar at� um estado terminal.
   */
  static int TAM = 3, PROF = -1;
  
  public static void main (String[] args)
  {
    Scanner ent = new Scanner (System.in);
    //Objeto da Classe Tabuleiro
    if(args.length > 0){
	TAM = Integer.parseInt(args[0]);
    }
    Tabuleiro t = new Tabuleiro (TAM);
    //Objeto da Classe Minimax
    MiniMax mm = new MiniMax (TAM, PROF);
    System.out.println("J2VELHA\nBem vindo ao Jogo!\nBoa Sorte!\n\n");
    //Imprime o tabuleiro na Tela
    t.imprimir ();
    do
    {  //Captura jogada do usu�rio
      int l, c;
      System.out.printf ("Sua jogada:\r\nLinha [0 - %d]: ", (TAM-1));
      l = ent.nextInt ();
      System.out.printf ("Coluna [0 - %d]: ", (TAM-1));
      c = ent.nextInt ();
      //Realiza jogada do usu�rio
      t.fazerJogada(l, c);
      t.imprimir ();
      //Verifica se n�o � um estado terminal
      if (!mm.teste_terminal(t.tabuleiro)) 
      {
        //Aplica o algoritmo minimax ao tabuleiro           
        t.tabuleiro = mm.decisao_minimax(t.tabuleiro);
        System.out.println ("Jogada do Computador:");
        t.imprimir ();
      }
    } while (!mm.teste_terminal(t.tabuleiro));
   //Verifica o ganhador, ou um empate 
    if (mm.ganhou(t.tabuleiro, 1))
      System.out.println("O computador ganhou!");
    else if (mm.ganhou(t.tabuleiro, -1))
      System.out.println("Voc� ganhou!");
    else
      System.out.println("Empate!");           
  }
}

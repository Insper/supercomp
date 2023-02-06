public class Tabuleiro {
    
    static char[] conversao = {'O', ' ', 'X'};
    static int[][] tabuleiro;
    int tam;
    String divisor;
    
    public  Tabuleiro (int tam){
        this.tam = tam;
        tabuleiro = new int[tam][tam];
        divisor = gerarDivisor();
    }
    
    public void fazerJogada(int l, int c){
        if(tabuleiro[l][c] == 0)
            tabuleiro[l][c] = -1;
        else
            System.out.println("Posição ja sendo usada");
        }
    
    public void imprimir(){
        for(int i = 0; i < tam; i++){
            for(int j = 0; j < tam; j++){
                System.out.printf(" %c %c", conversao[tabuleiro[i][j] + 1], j == tam-1 ? ' ' : '|');
            }
        if(i != (tam-1))
            System.out.println(divisor);
        }
        System.out.println("\r\n");
    }
    
    public String gerarDivisor(){
        String d = "\r\n";
        
        for(int i = 0; i< (tam-1); i++){
            d+= "---+";
        }
        d+= "---";
        return d;
    }
}


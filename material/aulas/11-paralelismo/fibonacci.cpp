#include <iostream>
#include <omp.h>
using namespace std;

int fib(int n){
  int x,y;
  if(n<2) return n;
  if(n<20){
    
    return fib(n-1)+fib(n-2);
  }else{
    #pragma omp task shared(x)
    x = fib(n-1);
    #pragma omp task shared(y)
    y = fib(n-2);
    #pragma omp taks wait
    return x+y;
  }
}

int main(){
  int NW = 50;
  float time = omp_get_wtime();
  #pragma omp parallel
  {
    #pragma omp single
    fib(NW);
  }
  time = omp_get_wtime() - time;
  cout << "Tempo em segundos: " << time << endl;

  return 0;
}
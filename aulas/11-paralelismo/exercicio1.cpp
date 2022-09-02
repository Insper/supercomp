#include <iostream>
#include <unistd.h>

double funcao1() {
    sleep(4);
    return 47;
}

double funcao2() {
    sleep(1);
    return -11.5;
}

int main() {
    double res_func1, res_func2;

    res_func1 = funcao1();
    res_func2 = funcao2();

    std::cout << res_func1 << " " << res_func2 << "\n";
}

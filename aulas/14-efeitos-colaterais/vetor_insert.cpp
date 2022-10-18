#include <vector>
#include <iostream>
#include <unistd.h>


double conta_complexa(int i) {
	return 2 * i;
}

int main() {
	int N = 10000; 
	std::vector<double> vec;
	for (int i = 0; i < N; i++) {
		vec.push_back(conta_complexa(i));
	}
	
	for (int i = 0; i < N; i++) {
		std::cout << i << " ";
	}
	
	return 0;
}

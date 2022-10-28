/* knapsack heuristica*/
#include <iostream>
#include <algorithm>    // std::sort
#include <vector>       // std::vector

// Estrutura básica para objetos
struct Object {
  int id;
  int weight;
  int value;
};

// Ordena objetos
bool sort_object(Object i, Object j) { return (i.value < j.value); }
 

int main() {

    int n, W;

    std::cin >> n; // quantidade de elementos
    std::cin >> W; // capacidade da mochila

    std::vector<Object> objects;
    std::vector<Object> bag;

    for(int i=0; i < n; i++) {
        Object tmp;
        tmp.id = i;
        std::cin >> tmp.weight;
        std::cin >> tmp.value;
        objects.push_back(tmp);
    }

    std::sort(objects.begin(), objects.end(), sort_object);

    int inside_bag = 0;
    int total_value = 0;

    for(int i=n-1; i>=0; i--) {
        std::cout << "(" << objects[i].id << " " << objects[i].weight << " " << objects[i].value << ") ";
        if(objects[i].weight <= W - inside_bag) {
            bag.push_back(objects[i]);
            inside_bag += objects[i].weight;
            total_value += objects[i].value;
            objects.erase(objects.begin()+i);
        }
    }
    std::cout << std::endl;

    std::cout << inside_bag << " " << total_value << " 0" << std::endl;
    for(Object o : bag) {
        std::cout << o.id << " ";
    }
    std::cout << std::endl;

    return 0;
}
 
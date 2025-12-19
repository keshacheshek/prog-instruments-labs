#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main() {
    srand(time(0));
    ofstream file("cpp_sequence.txt");

    for (int i = 0; i < 128; i++) {
        int number = rand() % 2;
        cout << number;
        file << number;
    }

    file.close();
    cout << "\nSaved to Cpp_Sequence.txt";
    return 0;
}
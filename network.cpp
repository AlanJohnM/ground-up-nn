//
//  network.cpp
//  
//
//  Created by Alan Milligan on 2019-12-12.
//

#include <stdio.h>
#include <iostream>
#include <random>
#include <cmath>
#include "eigen-3.3.7/Eigen/Dense"



using namespace std;

class Network
{
    
private:
    
    
    
    
    
    
    void init(int layers, int sizes[]) {
        
    }
    
    void feedForward() {
        
    }
    
    
    void backProp() {
        
    }
    
    double ReLu(double x) {
        return max(x,0);
    }
    
    double ReLuPrime(double x) {
        return (x > 0 ? 1 : );
    }
    
    
public:
    
    Network(int layers, int sizes[]) {
        init(layers,sizes);
    }
    
    
    void train() {
        
    }
    
    void test() {
        
    }
};

int main() {
    
    cout << "test" << endl;
}



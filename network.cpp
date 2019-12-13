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
using namespace Eigen;

class Network
{
    
private:
    
    VectorXd* ActivationLayers;
    VectorXd* ActivationPrimeLayers;
    MatrixXd* Weights;
    int layerCount;
    double learnRate;
    
    
    void init(int layerCount, int sizes[]) {
        this->layerCount = layerCount;
        ActivationLayers = new VectorXd[layerCount];//remember to delete
        ActivationPrimeLayers = new VectorXd[layerCount];//remember to delete
        Weights = new MatrixXd[layerCount-1];//remember to delete
        for (int i = 0; i < layerCount; i++) {
            ActivationLayers[i] = VectorXd(sizes[i]);
            ActivationPrimeLayers[i] = VectorXd(sizes[i]);
            ActivationLayers[i].setZero();
            ActivationPrimeLayers[i].setZero();
        }
        for (int i = 0; i < layerCount-1; i++) {
            Weights[i] = MatrixXd(sizes[i+1],sizes[i]);
            Weights[i].setRandom();
            Weights[i] *= 100;
        }
    }
    
    //assumes input is already loaded into ActivationLayers[0]
    void feedForward() {
        for (int i = 1; i < layerCount; i++) {
            VectorXd v(Weights[i-1]*ActivationLayers[i-1]);
            ActivationLayers[i] = v;
            ActivationPrimeLayers[i] = v;
            //could be replaced with different Activation functions here
            ReLu(&ActivationLayers[i]);
            ReLuPrime(&ActivationPrimeLayers[i]);
        }
//        cout << ActivationLayers[layerCount-1] << endl;
    }
    
    //currently naive GD, will need to change to SGD or batch or something
    //assumes output to be tested is loaded into ActivationLayers[layerCount-1]
    void backProp(VectorXd desired) {
        VectorXd e = ErrorFunctionPrime(ActivationLayers[layerCount-1],desired);
        VectorXd delta = e.cwiseProduct(ActivationPrimeLayers[layerCount-1]);
        VectorXd nextDelta;
        for (int i = layerCount-2; i >= 0; i--) {
            nextDelta = (Weights[i].transpose()*delta).cwiseProduct(ActivationPrimeLayers[i]);
            Weights[i] -= learnRate*delta*ActivationLayers[i].transpose();
            delta = nextDelta;
        }
    }
    
    //maybe see if i can define a new entrywise operation
    void ReLu(VectorXd* v) {
        for (int i = 0; i < v->rows(); i++) {
            (*v)[i] = fmax(0,(*v)[i]);
        }
    }
    
    //maybe see if i can define a new entrywise operation
    void ReLuPrime(VectorXd* v) {
        for (int i = 0; i < v->rows(); i++) {
          (*v)[i] = ((*v)[i] >= 0 ? 1 : 0);
        }
    }
    
    //abstracted out in case of more complex functions
    VectorXd ErrorFunctionPrime(VectorXd output, VectorXd desired) {
        return output - desired;
    }
    
    
    void logState() {
        cout << "CURRENT WEIGHTS:" << endl;
        for (int i = layerCount-2; i >= 0; i--) {
            cout << Weights[i] << endl;
        }
        cout << endl;
        cout << "CURRENT ACTIVATIONS:" << endl;
        for (int i = layerCount-1; i >= 0; i--) {
            cout << ActivationLayers[i] << endl << endl;
        }
        cout << endl;
        cout << "CURRENT ACTIVATIONSPRIME:" << endl;
        for (int i = layerCount-1; i >= 0; i--) {
            cout << ActivationPrimeLayers[i] << endl<< endl;
        }
    }
    


    
public:
    
    Network(int layerCount, int sizes[]) {
        init(layerCount,sizes);
    }
    
    //write state to some sort of file, be able to load a pre-computed state
    void train(double learnRate) {
        this->learnRate = learnRate;
        VectorXd v(15);
        v << 100,100,100,100,0,0,0,100,100,0,100,100,100,100,0;
        VectorXd t(5);
        t << 1,0,0,0,0;
        
        testCase(v,t);
    }
    
    void test() {
        
    }
    
    void testCase(VectorXd in, VectorXd t) {
        ActivationLayers[0] = in;
        ActivationPrimeLayers[0].setOnes();
        feedForward();
        logState();
        backProp(t);
        logState();
    }
};

int main() {
    Network* t = new Network(4,(int[]){15,10,7,5});
    
    t->train(0.01);
    
    delete t;
}



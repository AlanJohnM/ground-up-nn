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
#include "mnistReader.cpp"


#define CYCLES 11//cant go above 10?

#define TRAININGLABEL "train-labels.idx1-ubyte.gz"
#define TRAININGIMAGE "train-images.idx3-ubyte.gz"

#define TESTLABEL "t10k-labels.idx1-ubyte.gz"
#define TESTIMAGE "t10k-images.idx3-ubyte.gz"

using namespace std;
using namespace Eigen;

class Network
{

public:
    
    Network(int layerCount, int sizes[]) {
        init(layerCount,sizes);
    }
    
    ~Network() {
        delete ActivationLayers;
        delete ActivationPrimeLayers;
        delete Weights;
        delete reader;
    }
    
    //write state to some sort of file, be able to load a pre-computed state
    //might need to make this normalize the data, could probably do that in reader?
    void train(double learnRate) {
        this->learnRate = learnRate;
        unsigned char** input = reader->readImages(TRAININGIMAGE);
        unsigned char* desired = reader->readLabels(TRAININGLABEL);
        ActivationPrimeLayers[0].setOnes();//probably wrong?
        
//        for (int i = 1; i <= 28*28; i++) {
//            cout << (double)input[1][i] << " ";
//            if (i%28==0){cout<<endl;}
//        }
        
        for (int i = 0; i < CYCLES; i++) {
            Map<VectorXd> in((double*)input[i],28*28);
            ActivationLayers[i] = in;
            VectorXd correct(10);
            correct.setZero();
            correct[desired[i]] = 1;
            feedForward();
            backProp(correct);
        }
        delete input;//this probably leaks mem
        delete desired;
    }
    
    void test() {
        
    }
    
    void testCase(VectorXd in, VectorXd t) {
        ActivationLayers[0] = in;
        ActivationPrimeLayers[0].setOnes();//????
        feedForward();
        //logState();
        backProp(t);
        for (int i = 0; i < 10000; i++) {
            feedForward();
            backProp(t);
        }
        //logState();
        feedForward();
        //logState();
    }
    
        
private:
        
    VectorXd* ActivationLayers;
    VectorXd* ActivationPrimeLayers;
    MatrixXd* Weights;
    int layerCount;
    double learnRate;
    mnistReader* reader;
    
    void init(int layerCount, int sizes[]) {
        this->layerCount = layerCount;
        reader = new mnistReader();
        ActivationLayers = new VectorXd[layerCount];
        ActivationPrimeLayers = new VectorXd[layerCount];
        Weights = new MatrixXd[layerCount-1];
        for (int i = 0; i < layerCount; i++) {
            ActivationLayers[i] = VectorXd(sizes[i]);
            ActivationPrimeLayers[i] = VectorXd(sizes[i]);
            ActivationLayers[i].setZero();
            ActivationPrimeLayers[i].setZero();
        }
        for (int i = 0; i < layerCount-1; i++) {
            Weights[i] = MatrixXd(sizes[i+1],sizes[i]);
            Weights[i].setRandom();
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
            //(*v)[i] = fmax(0,(*v)[i]);
            //THIS IS ACTUALLY LOGISITIC FOR NOW
            (*v)[i] = (1/(1+exp(-(*v)[i])));
        }
    }
    
    //maybe see if i can define a new entrywise operation
    void ReLuPrime(VectorXd* v) {
        for (int i = 0; i < v->rows(); i++) {
//          (*v)[i] = ((*v)[i] >= 0 ? 1 : 0);
            //THIS IS ACTUALLY LOGISITIC FOR NOW
            (*v)[i] = (1/(1+exp(-(*v)[i])))*(1-(1/(1+exp(-(*v)[i]))));
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
};

int main() {
    Network* t = new Network(4,(int[]){784,250,50,10});
    t->train(0.01);
    
    delete t;
}



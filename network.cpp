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

#define CYCLES 60000
#define TESTS 10000

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
    
    //write state to some sort of file, be able to load a pre-computed state
    //might need to make this normalize the data, could probably do that in reader?
    void train(double learnRate) {
        this->learnRate = learnRate;
        unsigned char** input = reader->readImages(TRAININGIMAGE);
        unsigned char* desired = reader->readLabels(TRAININGLABEL);
        ActivationPrimeLayers[0].setOnes();
        
        for (int i = 0; i < CYCLES; i++) {
            ActivationLayers[0] = vectorfy(input[i],28*28);
            VectorXd correct(10);
            correct.setZero();
            correct[desired[i]] = 1;
            feedForward();
            backProp(correct);
        }
        //do some memory managment here
    }
    

    void test() {
        unsigned char** input = reader->readImages(TESTIMAGE);
        unsigned char* desired = reader->readLabels(TESTLABEL);
        double totalError = 0;
        int good = 0;
        int bad = 0;
        
        for (int i = 0; i < TESTS; i++) {
            ActivationLayers[0] = vectorfy(input[i],28*28);
            VectorXd correct(10);
            correct.setZero();
            correct[desired[i]] = 1;
            feedForward();
            
            if (maxIndex(ActivationLayers[layerCount-1]) == desired[i]) {
                good++;
            } else {
                bad++;
            }
            
            cout << ActivationLayers[layerCount-1] << endl << correct << endl;
            cout << "E:" << ErrorFunction(ActivationLayers[layerCount-1],correct) << endl << endl;
            totalError += ErrorFunction(ActivationLayers[layerCount-1],correct);
        }
    
        double avgError = totalError/TESTS;
        cout << "AVERAGE ERROR: " << avgError << endl;
        cout << "GOOD: " << good << endl;
        cout << "BAD: " << bad << endl;
    }
    
    void testCase(VectorXd in, VectorXd t) {

    }
    
private:
        
    VectorXd* ActivationLayers;
    VectorXd* ActivationPrimeLayers;
    MatrixXd* Weights;
    mnistReader* reader;
    int layerCount;
    double learnRate;
    
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
            ActivationFunction(&ActivationLayers[i]);
            ActivationFunctionPrime(&ActivationPrimeLayers[i]);
        }
    }
    
    //update to use weights as well, maybe try batch instead of SGD
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
    
    //also normalizes
    VectorXd vectorfy(unsigned char* input, int size) {
        double arr[size];
        for (int i = 0; i < size; i++) {
            arr[i] = ((double)input[i]/128)-1;
        }
        Map<VectorXd> v(arr,size);
        return v;
    }
    
    int maxIndex(VectorXd v) {
        double maxVal = 0;
        int maxIdx = 0;
        for (int i = 0; i < v.rows(); i++) {
            if (v[i] > maxVal) {
                maxVal = v[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    void ActivationFunction(VectorXd* v) {
        for (int i = 0; i < v->rows(); i++) {
//            (*v)[i] = fmax(0,(*v)[i]);
            (*v)[i] = (1/(1+exp(-(*v)[i])));
        }
    }
    
    void ActivationFunctionPrime(VectorXd* v) {
        for (int i = 0; i < v->rows(); i++) {
//          (*v)[i] = ((*v)[i] >= 0 ? 1 : 0);
            (*v)[i] = (1/(1+exp(-(*v)[i])))*(1-(1/(1+exp(-(*v)[i]))));
        }
    }
    
    double ErrorFunction(VectorXd out, VectorXd desired){
        VectorXd diff = out - desired;
        diff = diff.array().pow(2);
        double err = 0;
        for (int i = 0; i < 10; i++) {
            err += diff[i]/2;
        }
        return err;
    }
    
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
    
    void printer(unsigned char* pic) {
        for (int i = 1; i <= 28*28; i++) {
            cout << (double)pic[i] << " ";
            if (i%28==0){cout<<endl;}
        }
    }
    
};

int main() {
    Network* t = new Network(5,(int[]){784,256,128,64,10});
    t->train(0.1);
    t->train(0.1);
    t->train(0.1);
    t->train(0.1);
    t->train(0.1);
    t->train(0.1);
    t->train(0.1);
    t->train(0.1);
    t->train(0.01);
    t->train(0.01);
    t->train(0.01);
    t->train(0.01);
    t->train(0.01);
    t->train(0.01);
    t->train(0.01);
    t->train(0.01);
    t->train(0.01);
    t->train(0.01);
    t->train(0.01);
    t->train(0.01);
    t->train(0.01);
    t->train(0.01);
    t->train(0.01);
    t->train(0.01);
    t->test();
    
    delete t;
}

//
//  mnistReader.cpp
//  
//
//  Created by Alan Milligan on 2019-12-13.
//

#include <stdio.h>
#include <iostream>
#include <fstream>



using namespace std;

class mnistReader
{
    
public:
    
    mnistReader() {}
    
    unsigned char* readLabels(char path[]) {
        ifstream raw(path,ifstream::binary);
        int magic;
        raw.read((char*)&magic,4);
        int count;
        raw.read((char*)&count,4);
        count = endianSwap(count);
        
        unsigned char* arr = new unsigned char[count];
        raw.read((char*)arr,count);
        return arr;
    }
    
    unsigned char** readImages(char path[]) {
        ifstream raw(path,ifstream::binary);
        int magic;
        int count;
        int rows;
        int cols;
        raw.read((char*)&magic,4);
        raw.read((char*)&count,4);
        raw.read((char*)&rows,4);
        raw.read((char*)&cols,4);
        magic = endianSwap(magic);
        count = endianSwap(count);
        rows = endianSwap(rows);
        cols = endianSwap(cols);
        
        unsigned char** mat = new unsigned char*[count];
        for (int i = 0; i < count; i++) {
            mat[i] = new unsigned char[rows*cols];
            raw.read((char*)(mat[i]),rows*cols);
        }
        return mat;
    }
    
private:
    int endianSwap(int i) {
        return (i >> 24 & 0x000000ff)
                +(i >> 8 & 0x0000ff00)
                +(i << 8 & 0x00ff0000)
                +(i << 24 & 0xff000000);
    }
    
};


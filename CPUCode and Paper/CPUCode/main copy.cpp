//
//  main.cpp
//  alf
//
//  Created by Guillaume GALES on 02/05/2014.
//  Copyright (c) 2014 Guillaume Gales. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

#include "Vec.h"
#include "Tools.h"
#include "Alf.h"
#include "Hist.h"



void process(const char *in, const char *bck, const char* gt,
             int method, int binSize, int size, double backgroundDistance, double distTh,
             const char *outA, const char *outAF){
    
    std::cout << method;
    std::cout << "\t" << binSize;
    std::cout << "\t" << size;
    std::cout << "\t" << backgroundDistance;
    std::cout << "\t" << distTh;

    
    //Input
    GG::Vec backgroundColor(60.0,-66.0,45.0);   //LCD green
    
    
    cv::Mat input = cv::imread(in, CV_LOAD_IMAGE_COLOR);
    input = GG::Tools::RGBtoLab(input);
    cv::Mat back = cv::imread(bck, CV_LOAD_IMAGE_COLOR);
    back = GG::Tools::RGBtoLab(back);
    //GT
    cv::Mat GTAF = cv::imread(gt, CV_LOAD_IMAGE_COLOR);
    GTAF=GG::Tools::RGBtoLab(GTAF);
    
    //Histogram
    GG::Hist h(input, binSize);
    std::list<GG::Vec> clusters=h.meanShift(size);
    GG::Hist::removeBackgroundClusters(clusters, backgroundColor, backgroundDistance);

    
    //Alf
    cv::Mat F=cv::Mat(input.rows,input.cols,CV_32FC3);
    cv::Mat AF=cv::Mat(input.rows,input.cols,CV_32FC3);
//    cv::Mat A=GG::Alf::goAlf(input, back, clusters, backgroundDistance, wwidth, p, F, AF);
    cv::Mat A=GG::Alf::goAlf0(input, back, clusters, backgroundDistance, distTh, F, AF, method);//, GTF);
    
    //Eval
    double error = GG::Alf::eval(AF,GTAF);
    std::cout << "\t" << error << "\n";
    
    //Alpha to RGB16
    A = GG::Tools::Alpha2RGB16(A);
//    F = GG::Tools::Lab2RGB16(F);
    AF = GG::Tools::Lab2RGB16(AF);

    cv::imwrite(outA, A);
    cv::imwrite(outAF, AF);
    
}

int main(int argc, const char * argv[])
{

    std::cout << "Alf\n";
    
    
    const char *input = argv[1];
    const char *back = argv[2];
    const char *gt = argv[3];
    int method = atoi(argv[4]);
    int binSize = atoi(argv[5]);
    int size = atoi(argv[6]);
    double backgroundDistance = atof(argv[7]);
    double distTh = atof(argv[8]);
    const char *outA = argv[9];
    const char *outAF = argv[10];
    
    
    process(input, back, gt, method, binSize, size, backgroundDistance, distTh, outA, outAF);
    
    
    std::cout << "xoxo\n";
    
    return 0;
}


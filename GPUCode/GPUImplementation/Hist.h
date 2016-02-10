//
//  Hist.h
//  alf
//
//  Created by Guillaume GALES on 02/05/2014.
//  Copyright (c) 2014 Guillaume Gales. All rights reserved.
//

#ifndef __alf__Hist__
#define __alf__Hist__

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Vec.h"
#include<stdio.h>
#include "histo.h"

namespace GG {

    /**
     * 3D Histogram and mean shift clustering
     */
    class Hist {

    private:
        int *count = NULL;

        int findBinIdxForDim(float val, int low){
            return ((int)round(val)-low)/binSize;
        }

        int findIdx(int idx1, int idx2, int idx3){
            return (idx1*nbBinsPerDim*nbBinsPerDim)+idx2*nbBinsPerDim+idx3;
        }

        void startCounting(cv::Mat &img, int nbBinsPerDim, int binSize){
        	int imgRows = img.rows;
        	int imgCols = img.cols;
        	int imgSize = imgRows*imgCols;
        	float *imgDataBuffer = NULL;
        	int imgLength = imgRows*imgCols*3*sizeof(float);
        	int imgDataBufferSize = imgRows*imgCols*3;
        	imgDataBuffer = (float*) malloc(imgLength);
            memset(imgDataBuffer, 0, imgLength);
        	int imgDataIndex = 0;
            for(int i=0; i<img.rows; i++){
                float *ptr = (float*) (img.data + i*img.step);
                for(int j=0; j<img.cols; j++){
                    imgDataBuffer[imgDataIndex] = ptr[j*3];
                    imgDataIndex++;
                    imgDataBuffer[imgDataIndex] = ptr[j*3+1];
                    imgDataIndex++;
                    imgDataBuffer[imgDataIndex] = ptr[j*3+2];
                    if(imgDataIndex != (imgDataBufferSize - 1))
                    	imgDataIndex++;

                }
            }
createHisto(imgDataBuffer, imgDataBufferSize, nbBinsPerDim, imgSize, binSize, count);
FILE *histo;
FILE *histoNonZero;
histo = fopen("/home/anirudh/Desktop/histoOutputG.txt", "a");
histoNonZero = fopen("/home/anirudh/Desktop/histoNonZeroG.txt", "a");
for(int i = 0; i < nbBinsPerDim * nbBinsPerDim * nbBinsPerDim; i++){
	fprintf(histo, "%d) %d \n",i+1, count[i]);
	if(count[i] != 0)
	fprintf(histoNonZero, "%d) %d \n",i+1, count[i]);
}
fclose(histo);
fclose(histoNonZero);
        }

        std::list<GG::Vec> initList() {
            std::list<GG::Vec> l;
            int offset = binSize/2;
            for (int L=0; L<=100; L+=binSize){
                for (int a=-127; a<=127; a+=binSize){
                    for (int b=-127; b<=127; b+=binSize){
                        int idx = findIdx(findBinIdxForDim(L, 0), findBinIdxForDim(a, -127), findBinIdxForDim(b, -127));
                        int c=count[idx];
                        if (c>0){
                            //Add new item to the list (use center of bin)
                            GG::Vec p(L+offset,a+offset,b+offset,c);
                            l.push_back(p);
                        }
                    }
                }
            }
            return l;
        }

    public:

        int binSize;
        int nbBinsPerDim;

        //Constructor
        Hist(cv::Mat &img, int binSize){
            this->binSize=binSize;
            //Init count
            nbBinsPerDim = 255/binSize;
            int length = nbBinsPerDim * nbBinsPerDim * nbBinsPerDim * sizeof(int);
            count = (int*) malloc(length);
            memset(count, 0, length);
            //Start counting
            this->startCounting(img, nbBinsPerDim, binSize);
        }

        //Destructor
        ~Hist(){
            if(count!=NULL)
                free(count);
        }

        //Mean shift member function
        std::list<GG::Vec> meanShift(int size) {

            //Init list
            std::list<GG::Vec> lA = initList();
            std::list<GG::Vec> lB;

            int iter=0;
            unsigned char cpt;
            do {

                cpt=0;
                while (!lA.empty()) {

                    //Pop first element
                    GG::Vec cA = lA.front();
                    lA.pop_front();

                    //Get all elements within the distance
                    int mL = cA.w*(int)round(cA.x);
                    int ma = cA.w*(int)round(cA.y);
                    int mb = cA.w*(int)round(cA.z);
                    int sum = cA.w;

                    std::list<GG::Vec>::iterator it = lA.begin();

                    int k=0;
                    while(it!=lA.end()){
                        GG::Vec cB = *it;
                        GG::Vec BA= cB-cA;
                        int dist = (int) round(BA.norm2());

                        if (dist <= (size*binSize)*(size*binSize)){
                            mL+=cB.w*(int)round(cB.x);
                            ma+=cB.w*(int)round(cB.y);
                            mb+=cB.w*(int)round(cB.z);
                            sum+=cB.w;
                            //Remove this element
                            it=lA.erase(it);
                            cpt++;
                            k++;
                        } else {
                            it++;
                        }
                    }//end while

                    mL/=sum;
                    ma/=sum;
                    mb/=sum;
                    GG::Vec mc(mL,ma,mb,sum);

                    lB.push_back(mc);

                }//end while list not empty

                lA = lB;
                lB.clear();
                iter++;

            }while (cpt>0);

            //Sort my list per decreasing n
            lA.sort(GG::Vec::wComp);


//            lA.push_back(C(0,0,0,1));//add black
//            lA.push_back(C(100,0,0,1));//add white
//            lA.push_back(C(50,0,0,1));//add greys
//            lA.push_back(C(25,0,0,1));
//            lA.push_back(C(75,0,0,1));


            return lA;

        }//end mean shift


        static void removeBackgroundClusters(std::list<GG::Vec> &lA, GG::Vec backgroundColour,
                                             double backgroundDistance){

            //Remove background clusters
            std::list<GG::Vec>::iterator it = lA.begin();
            while(it!=lA.end()){
                GG::Vec c=*it;
                GG::Vec BC=backgroundColour-c;
              double dist = BC.norm2();
//                double dist = (BC.y*BC.y+BC.z*BC.z);   //chroma only
                if(dist<=backgroundDistance*backgroundDistance){
                    it=lA.erase(it);
                } else {
                    it++;
                }
            }
        }

        static void removeBackgroundClusters2(std::list<GG::Vec> &lA, GG::Vec backgroundColour,
                                             double backgroundDistance){

            double hardTh = backgroundDistance*0.75;

            //Remove background clusters
            std::list<GG::Vec>::iterator it = lA.begin();
            while(it!=lA.end()){
                GG::Vec c=*it;
                GG::Vec BC=backgroundColour-c;
                double dist = BC.norm2();
                //                double dist = (BC.y*BC.y+BC.z*BC.z);   //chroma only
                if(dist<=hardTh*hardTh){
                    it=lA.erase(it);
                } else {

                    //Elastic Push current cluster toward boundary
                    //?

                    it++;
                }
            }
        }

    }; //end class

} //end namespace

#endif /* defined(__alf__Hist__) */

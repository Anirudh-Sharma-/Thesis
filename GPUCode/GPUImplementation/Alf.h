//
//  Alf.h
//  alf
//
//  Created by Guillaume GALES on 02/05/2014.
//  Copyright (c) 2014 Guillaume Gales. All rights reserved.
//

#ifndef __alf__Alf__
#define __alf__Alf__

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Vec.h"

namespace GG {

    /**
     * Provides tools for image processing (requires OpenCV)
     */
    class Alf {

    public:
        static cv::Mat groundTruthAlpha(cv::Mat &O1, cv::Mat &O2, cv::Mat &B1, cv::Mat &B2,
                                        cv::Mat &F, cv::Mat &AF){

            cv::Mat A=cv::Mat::zeros(O1.rows, O1.cols, CV_32FC1);

            for(int i=0;i<O1.rows;i++){

                float *pO1 = (float*) (O1.data+i*O1.step);
                float *pO2 = (float*) (O2.data+i*O2.step);
                float *pB1 = (float*) (B1.data+i*B1.step);
                float *pB2 = (float*) (B2.data+i*B2.step);
                float *pA = (float*) (A.data+i*A.step);
                float *pF = (float*) (F.data+i*F.step);
                float *pAF = (float*) (AF.data+i*AF.step);

                for(int j=0;j<O1.cols;j++){
                    GG::Vec O1(pO1,j);
                    GG::Vec O2(pO2,j);
                    GG::Vec B1(pB1,j);
                    GG::Vec B2(pB2,j);
                    GG::Vec O1O2=O2-O1;
                    GG::Vec B1B2=B2-B1;
                    double alpha = 1.0 - (GG::Vec::dot(O1O2, B1B2) / B1B2.norm2());
                    alpha=(alpha>1.0)?1.0:alpha;    //Foreground
                    GG::Vec f;
                    if(alpha<0){
                        //Switch B and O
                        alpha = 1.0 - (GG::Vec::dot(O1O2, B1B2) / O1O2.norm2());
                        f=(B1-((1.0-alpha)*O1))*(1.0/alpha);
                    } else {
                        f=(O1-((1.0-alpha)*B1))*(1.0/alpha);

                    }
                    pA[j]=(float)alpha;
                    pF[j*3]=(float)f.x;
                    pF[j*3+1]=(float)f.y;
                    pF[j*3+2]=(float)f.z;
                    pAF[j*3]=(float)(alpha*f.x);
                    pAF[j*3+1]=(float)(alpha*f.y);
                    pAF[j*3+2]=(float)(alpha*f.z);
                }
            }

            return A;

        }


        static cv::Mat groundTruthAlpha2(cv::Mat &b, cv::Mat &g, cv::Mat &k, cv::Mat &y, cv::Mat &r,
                                         cv::Mat &bb, cv::Mat &bg, cv::Mat &bk, cv::Mat &by, cv::Mat &br,
                                        cv::Mat &F, cv::Mat &AF){

            cv::Mat A=cv::Mat::zeros(b.rows, b.cols, CV_32FC1);

            cv::Mat mA = cv::Mat::zeros(15,4, CV_64FC1);
            cv::Mat mb = cv::Mat::zeros(15,1, CV_64FC1);
            cv::Mat mx = cv::Mat::zeros(4,1, CV_64FC1);

            for(int i=0;i<b.rows;i++){

                float *pb = (float*) (b.data+i*b.step);
                float *pg = (float*) (g.data+i*g.step);
                float *pk = (float*) (k.data+i*k.step);
                float *py = (float*) (y.data+i*y.step);
                float *pr = (float*) (r.data+i*r.step);
                float *pbb = (float*) (bb.data+i*bb.step);
                float *pbg = (float*) (bg.data+i*bg.step);
                float *pbk = (float*) (bk.data+i*bk.step);
                float *pby = (float*) (by.data+i*by.step);
                float *pbr = (float*) (br.data+i*br.step);


                float *pA = (float*) (A.data+i*A.step);
                float *pF = (float*) (F.data+i*F.step);
                float *pAF = (float*) (AF.data+i*AF.step);

                for(int j=0;j<b.cols;j++){

                    GG::Vec B(pb,j);
                    GG::Vec G(pg,j);
                    GG::Vec K(pk,j);
                    GG::Vec Y(py,j);
                    GG::Vec R(pr,j);
                    GG::Vec BB(pbb,j);
                    GG::Vec BG(pbg,j);
                    GG::Vec BK(pbk,j);
                    GG::Vec BY(pby,j);
                    GG::Vec BR(pbr,j);

                    if((i==1800 && j==1520)||(i==480 && j==1860)){
                        std::cout << "i=" << i << " j=" << j << "\n";
                        std::cout << "B=" << B << "\n";
                        std::cout << "G=" << G << "\n";
                        std::cout << "K=" << K << "\n";
                        std::cout << "Y=" << Y << "\n";
                        std::cout << "R=" << R << "\n";
                        std::cout << "BB=" << BB << "\n";
                        std::cout << "BG=" << BG << "\n";
                        std::cout << "BK=" << BK << "\n";
                        std::cout << "BY=" << BY << "\n";
                        std::cout << "BR=" << BR << "\n";

                     }

                    //mA
                    mA.at<double>(0,0) = -BB.x; mA.at<double>(0,1) = 1.0;
                    mA.at<double>(1,0) = -BB.y; mA.at<double>(1,2) = 1.0;
                    mA.at<double>(2,0) = -BB.z; mA.at<double>(2,3) = 1.0;
                    mA.at<double>(3,0) = -BG.x; mA.at<double>(3,1) = 1.0;
                    mA.at<double>(4,0) = -BG.y; mA.at<double>(4,3) = 1.0;//!!!!!!!!!!!
                    mA.at<double>(5,0) = -BG.z; mA.at<double>(5,3) = 1.0;
                    mA.at<double>(6,0) = -BK.x; mA.at<double>(6,1) = 1.0;
                    mA.at<double>(7,0) = -BK.y; mA.at<double>(7,2) = 1.0;
                    mA.at<double>(8,0) = -BK.z; mA.at<double>(8,3) = 1.0;
                    mA.at<double>(9,0) = -BY.x; mA.at<double>(9,1) = 1.0;
                    mA.at<double>(10,0) = -BY.y; mA.at<double>(10,2) = 1.0;
                    mA.at<double>(11,0) = -BY.z; mA.at<double>(11,3) = 1.0;
                    mA.at<double>(12,0) = -BR.x; mA.at<double>(12,1) = 1.0;
                    mA.at<double>(13,0) = -BR.y; mA.at<double>(13,2) = 1.0;
                    mA.at<double>(14,0) = -BR.z; mA.at<double>(14,3) = 1.0;

                    //mb
                    mb.at<double>(0,0) = B.x - BB.x;
                    mb.at<double>(1,0) = B.y - BB.y;
                    mb.at<double>(2,0) = B.z - BB.z;
                    mb.at<double>(3,0) = G.x - BG.x;
                    mb.at<double>(4,0) = G.y - BG.y;
                    mb.at<double>(5,0) = G.z - BG.z;
                    mb.at<double>(6,0) = K.x - BK.x;
                    mb.at<double>(7,0) = K.y - BK.y;
                    mb.at<double>(8,0) = K.z - BK.z;
                    mb.at<double>(9,0) = Y.x - BY.x;
                    mb.at<double>(10,0) = Y.y - BY.y;
                    mb.at<double>(11,0) = Y.z - BY.z;
                    mb.at<double>(12,0) = R.x - BR.x;
                    mb.at<double>(13,0) = R.y - BR.y;
                    mb.at<double>(14,0) = R.z - BR.z;

                    if((i==1800 && j==1520)||(i==480 && j==1860)){
                    std::cout << "A=[\n";
                    for(int k=0;k<15;k++){
                        for(int l=0;l<4;l++){
                            std::cout << mA.at<double>(k,l) << " ";
                        }
                        std::cout << "\n";
                    }
                    std::cout <<"];\n";

                    std::cout << "b=[\n";
                    for(int k=0;k<15;k++){
                        for(int l=0;l<1;l++){
                            std::cout << mb.at<double>(k,l) << " ";
                        }
                        std::cout << "\n";
                    }
                    std::cout <<"];\n";
                    }
                    //mx
                    cv::solve(mA,mb,mx,cv::DECOMP_QR);

                    if((i==1800 && j==1520)||(i==480 && j==1860)){
                    std::cout << "x=[\n";
                    for(int k=0;k<4;k++){
                        for(int l=0;l<1;l++){
                            std::cout << mx.at<double>(k,l) << " ";
                        }
                        std::cout << "\n";
                    }
                    std::cout <<"];\n";
                    }

                    double alpha=mx.at<double>(0,0);
                    GG::Vec f(mx.at<double>(1,0),mx.at<double>(2,0),mx.at<double>(3,0));

//                    if(alpha<0){
//                        //Swap B<->O
//                        //mA
//                        mA.at<double>(0,0) = -B.x; mA.at<double>(0,1) = 1.0;
//                        mA.at<double>(1,0) = -B.y; mA.at<double>(1,2) = 1.0;
//                        mA.at<double>(2,0) = -B.z; mA.at<double>(2,3) = 1.0;
//                        mA.at<double>(3,0) = -G.x; mA.at<double>(3,1) = 1.0;
//                        mA.at<double>(4,0) = -G.y; mA.at<double>(4,3) = 1.0;
//                        mA.at<double>(5,0) = -G.z; mA.at<double>(5,3) = 1.0;
//                        mA.at<double>(6,0) = -K.x; mA.at<double>(6,1) = 1.0;
//                        mA.at<double>(7,0) = -K.y; mA.at<double>(7,2) = 1.0;
//                        mA.at<double>(8,0) = -K.z; mA.at<double>(8,3) = 1.0;
//                        mA.at<double>(9,0) = -Y.x; mA.at<double>(9,1) = 1.0;
//                        mA.at<double>(10,0) = -Y.y; mA.at<double>(10,2) = 1.0;
//                        mA.at<double>(11,0) = -Y.z; mA.at<double>(11,3) = 1.0;
//                        mA.at<double>(12,0) = -R.x; mA.at<double>(12,1) = 1.0;
//                        mA.at<double>(13,0) = -R.y; mA.at<double>(13,2) = 1.0;
//                        mA.at<double>(14,0) = -R.z; mA.at<double>(14,3) = 1.0;
//
//                        //mb
//                        mb.at<double>(0,0) = BB.x - B.x;
//                        mb.at<double>(1,0) = BB.y - B.y;
//                        mb.at<double>(2,0) = BB.z - B.z;
//                        mb.at<double>(3,0) = BG.x - G.x;
//                        mb.at<double>(4,0) = BG.y - G.y;
//                        mb.at<double>(5,0) = BG.z - G.z;
//                        mb.at<double>(6,0) = BK.x - K.x;
//                        mb.at<double>(7,0) = BK.y - K.y;
//                        mb.at<double>(8,0) = BK.z - K.z;
//                        mb.at<double>(9,0) = BY.x - Y.x;
//                        mb.at<double>(10,0) = BY.y - Y.y;
//                        mb.at<double>(11,0) = BY.z - Y.z;
//                        mb.at<double>(12,0) = BR.x - R.x;
//                        mb.at<double>(13,0) = BR.y - R.y;
//                        mb.at<double>(14,0) = BR.z - R.z;
//
//
//                        //mx
//                        cv::solve(mA,mb,mx,cv::DECOMP_QR);
//                        alpha=mx.at<double>(0,0);
//                        f.x=(float)mx.at<double>(1,0);
//                        f.y=(float)mx.at<double>(2,0);
//                        f.z=(float)mx.at<double>(3,0);
//
//                    }

                    pA[j]=(float)alpha;
                    pAF[j*3]=(float)f.x;
                    pAF[j*3+1]=(float)f.y;
                    pAF[j*3+2]=(float)f.z;
                    pF[j*3]=(float)(f.x/alpha);
                    pF[j*3+1]=(float)(f.y/alpha);
                    pF[j*3+2]=(float)(f.z/alpha);
                }
            }

            return A;

        }


        static cv::Mat goAlf(cv::Mat &img, cv::Mat &back, std::list<GG::Vec> &clusters,
                             int backgroundDistance, double wwidth, double p,
                             cv::Mat &Fmap, cv::Mat &AF){

            cv::Mat A=cv::Mat::zeros(img.rows, img.cols, CV_32FC1);

            double alpha;
            GG::Vec K(0,0,0,0);
            GG::Vec argF;

            for(int i=0; i<img.rows; i++){

                float *ptr = (float*) (img.data + i*img.step);
                float *bptr = (float*) (back.data + i*back.step);
                float *pA = (float*) (A.data+i*A.step);
                float *pF = (float*) (Fmap.data+i*Fmap.step);
                float *pAF = (float*) (AF.data+i*AF.step);

                for(int j=0; j<img.cols; j++){

                    GG::Vec B(bptr,j);
                    GG::Vec O(ptr,j);


                    GG::Vec BO=O-B;
                    GG::Vec BOu=BO;
                    BOu.unit();

                    double BOn2 = BO.norm2();
                    double BOn = sqrt(BOn2);

                    if(i==300 && j==200){
                        std::cout << "i=" << i << " j=" << j << "\n";
                        std::cout << "\tO=" << O << " B=" << B << " BO=" << BO << "\n";
                        std::cout << "\tBOu=" << BOu << "\n";
                        std::cout << "\tBOn2=" << BOn2 << " BOn=" << BOn << "\n";
                    }

                    if (BOn==0){
                        //O==B -> Background
                        argF = K;
                        alpha = 0;
                    } else {
                        if(BOn>=(double)backgroundDistance){
                            //TODO this th must be the same as the one to remove clusters ?
                            //Foreground
                            argF=O;
                            alpha=1.0;
                        } else
                        {

                            //For each cluster
                            double minCFn2 = DBL_MAX;
                            double argBFn;

                            std::list<GG::Vec>::iterator it;
                            for(it=clusters.begin(); it!=clusters.end(); it++){

                                GG::Vec C = *it;

                                GG::Vec BC=C-B;
                                double BCn2 = BC.norm2();
                                double BCn = sqrt(BCn2);

                                double BFn = GG::Vec::dot(BC, BOu);



                                GG::Vec F;
                                if (BFn>=0){
//                                    //Orthogonal projection of C on BO
//                                    F=(BFn*BOu)+B;
                                    //Rotation of BC on BO
                                    F=(BCn*BOu)+B;
                                } else {
//                                    //Switch O/B
//                                    //Orthogonal projection of C on BO
//                                    F=(-BFn*BOu)+O;
                                    //Rotation of BC on BO
                                    F=(-BCn*BOu)+O;
                                }



                                //Norm CF
                                GG::Vec CF=F-C;
                                double CFn2 = CF.norm2();
                                //Or angle as cost
                                double absBFn = (BFn>=0)?BFn:-BFn;
                                double theta = acos(absBFn/BCn);

                                double cost=theta; //CFn2;
                                if(cost<minCFn2){
                                    minCFn2=cost;
                                    argF=F;
                                    argBFn=BFn;
                                }

                                if(i==300 && j==200 && sqrt(cost) <= 0.7){
                                    std::cout << "\t\tC=" << C << "\n";
                                    std::cout << "\t\tcost=" << (cost) << "\n";
                                    std::cout << "\t\tBC=" << BC << "\n";
                                    std::cout << "\t\tF=" << F << "\n\n";
                                }

                            }//end for each cluster

                            if(i==300 && j==200){
                                std::cout << "\tBestF=" << argF << "\n";
                                std::cout << "\tcost=" << (minCFn2) << "\n";
                            }

                            //If the best cluster is too far away, then use the intersection of the CS bound
                            if(minCFn2>(M_PI/12.0)){

                                //Intersection BO Face 1-6

                                double d;
                                GG::Vec P0,n, X, bestX;

                                //Face A
                                P0=GG::Vec(50,-128,0);
                                n=GG::Vec(0,1,0);
                                X=GG::Vec::interRayPlane(B, BOu, P0, n, d);
                                if (X.x>=0 && X.x<=100 && X.y>-128 && X.y<128 && X.z>-128 && X.z<128 && d>0) {
                                    bestX=X;
                                }
                                //Face B
                                P0=GG::Vec(50,0,127);
                                n=GG::Vec(0,0,-1);
                                X=GG::Vec::interRayPlane(B, BOu, P0, n, d);
                                if (X.x>=0 && X.x<=100 && X.y>-128 && X.y<128 && X.z>-128 && X.z<128 && d>0){
                                    bestX=X;
                                }
                                //Face C
                                P0=GG::Vec(50,128,0);
                                n=GG::Vec(0,-1,0);
                                X=GG::Vec::interRayPlane(B, BOu, P0, n, d);
                                if (X.x>=0 && X.x<=100 && X.y>-128 && X.y<128 && X.z>-128 && X.z<128 && d>0){
                                    bestX=X;
                                }
                                //Face D
                                P0=GG::Vec(50,0,-128);
                                n=GG::Vec(0,0,1);
                                X=GG::Vec::interRayPlane(B, BOu, P0, n, d);
                                if (X.x>=0 && X.x<=100 && X.y>-128 && X.y<128 && X.z>-128 && X.z<128 && d>0){
                                    bestX=X;
                                }
                                //Face E
                                P0=GG::Vec(100,0,0);
                                n=GG::Vec(-1,0,0);
                                X=GG::Vec::interRayPlane(B, BOu, P0, n, d);
                                if (X.x>=0 && X.x<=100 && X.y>-128 && X.y<128 && X.z>-128 && X.z<128 && d>0) {
                                    bestX=X;
                                }
                                //Face F
                                P0=GG::Vec(0,0,0);
                                n=GG::Vec(1,0,0);
                                X=GG::Vec::interRayPlane(B, BOu, P0, n, d);
                                if (X.x>=0 && X.x<=100 && X.y>-128 && X.y<128 && X.z>-128 && X.z<128 && d>0){
                                    bestX=X;
                                }
                                //

                                argF=bestX;

                            } //end if mincost too large


                            //F is mix F/O depending on BOn
                            //If BOn very large -> F=O
                            //else F=F
                            double thMax=backgroundDistance;
                            double thMin=backgroundDistance-wwidth;

                            if(BOn<thMin){
                                //argF<-argF
                            } else {
                                if (BOn>=thMax){
                                    //argF=O; already set
                                } else {
                                    //F mix F/O
                                    double x=BOn-thMin;
                                    double weigth = pow(x/wwidth,p)/(pow(x/wwidth,p)+pow(1.0-(x/wwidth),p));
                                    argF=weigth*O+(1.0-weigth)*argF;
                                }
                            }



                            //

                            //Clamp argF to colorspace
                            argF.x=(argF.x<0)?0:argF.x;
                            argF.x=(argF.x>100)?100:argF.x;
                            argF.y=(argF.y<-127)?-127:argF.y;
                            argF.y=(argF.y>127)?127:argF.y;
                            argF.z=(argF.z<-127)?-127:argF.z;
                            argF.z=(argF.z>127)?127:argF.z;
                            //


                            //Alpha as BO/BF (O B F must be aligned)
                            GG::Vec BFF=argF-B;
                            double BF2n= BFF.norm2();
                            BF2n=sqrt(BF2n);
                            alpha=(BOn/BF2n);
                            //

                        }

                    }//end else pure background

                    pA[j] = alpha;

                    pF[j*3]=argF.x;
                    pF[j*3+1] = argF.y;
                    pF[j*3+2] = argF.z;

                    pAF[j*3]=alpha*argF.x;
                    pAF[j*3+1] = alpha*argF.y;
                    pAF[j*3+2] = alpha*argF.z;

                }//end for j
            }//end for i

            return A;

        }//end alf


        static cv::Mat goAlf1(cv::Mat &img, cv::Mat &back, std::list<GG::Vec> &clusters,
                             int backgroundDistance,
                             cv::Mat &Fmap, cv::Mat &AF){

            cv::Mat A=cv::Mat::zeros(img.rows, img.cols, CV_32FC1);

            double alpha;
            GG::Vec K(0,0,0,0);
            GG::Vec argF;
            double argAlpha;

            for(int i=0; i<img.rows; i++){

                float *ptr = (float*) (img.data + i*img.step);
                float *bptr = (float*) (back.data + i*back.step);
                float *pA = (float*) (A.data+i*A.step);
                float *pF = (float*) (Fmap.data+i*Fmap.step);
                float *pAF = (float*) (AF.data+i*AF.step);

                for(int j=0; j<img.cols; j++){

                    GG::Vec B(bptr,j);
                    GG::Vec O(ptr,j);


                    GG::Vec BO=O-B;
                    GG::Vec BOu=BO;
                    BOu.unit();

                    double BOn2 = BO.norm2();
                    double BOn = sqrt(BOn2);


                    if (BOn==0){
                        //O==B -> Background
                        argF = K;
                        alpha = 0;
                    } else {
//                        if(BOn>=(double)backgroundDistance){
//                            //TODO this th must be the same as the one to remove clusters ?
//                            //Foreground
//                            argF=O;
//                            alpha=1.0;
//                        } else
                        {

                            //For each cluster
                            double minCost = DBL_MAX;


                            std::list<GG::Vec>::iterator it;
                            for(it=clusters.begin(); it!=clusters.end(); it++){

                                GG::Vec C = *it;

                                GG::Vec BC = C-B;
                                alpha = GG::Vec::dot(BO, BC)/BC.norm2();

                                GG::Vec r=(BC*alpha) - BO;
                                double rho=r.norm2();

                                if (rho>30.0*30.0) continue;

                                GG::Vec F=((BO*BC.norm2())*(1.0/GG::Vec::dot(BO,BC)))+B;

                                GG::Vec CF=F-C;
                                double costCF = CF.norm2();

                                GG::Vec OF=F-O;
                                double costOF = OF.norm2();

                                GG::Vec BF=F-B;
                                double costBF = BF.norm2();

                                double cost=costOF; //rho;

                                if(i==360 && j==400){
                                    std::cout << "i=" << i << " j=" << j << "\n";
                                    std::cout << "O=" << O << " C=" << C << " B=" << B << "\n";
                                    std::cout << "F=" << F << " cost(CF)=" << sqrt(costCF) << " cost(OF)=" << sqrt(costOF) <<  " cost(BO)=" << BOn << " cost(BF)=" << sqrt(costBF) << "\n";
                                    std::cout << "cost=" << sqrt(cost) << "\n";
                                }

                                if(cost<minCost){
                                    minCost=cost;
                                    argF=F;
                                    argAlpha=alpha;
                                }

                            }//end for each cluster

                            //

                            //Clamp argF to colorspace
                            argF.x=(argF.x<0)?0:argF.x;
                            argF.x=(argF.x>100)?100:argF.x;
                            argF.y=(argF.y<-127)?-127:argF.y;
                            argF.y=(argF.y>127)?127:argF.y;
                            argF.z=(argF.z<-127)?-127:argF.z;
                            argF.z=(argF.z>127)?127:argF.z;
                            //


                        }

                    }//end else pure background

                    pA[j] = argAlpha;

                    pF[j*3]=argF.x;
                    pF[j*3+1] = argF.y;
                    pF[j*3+2] = argF.z;

                    pAF[j*3]=argAlpha*argF.x;
                    pAF[j*3+1] = argAlpha*argF.y;
                    pAF[j*3+2] = argAlpha*argF.z;

                }//end for j
            }//end for i

            return A;

        }//end alf 1


        static cv::Mat goAlf2(cv::Mat &img, cv::Mat &back, std::list<GG::Vec> &clusters,
                              int backgroundDistance,
                              cv::Mat &Fmap, cv::Mat &AF){

            cv::Mat A=cv::Mat::zeros(img.rows, img.cols, CV_32FC1);

            double alpha;
            GG::Vec K(0,0,0,0);
            GG::Vec Y(98.0,-16.0,93.0,0);
            GG::Vec argF;
            double argAlpha;

            for(int i=0; i<img.rows; i++){

                float *ptr = (float*) (img.data + i*img.step);
                float *bptr = (float*) (back.data + i*back.step);
                float *pA = (float*) (A.data+i*A.step);
                float *pF = (float*) (Fmap.data+i*Fmap.step);
                float *pAF = (float*) (AF.data+i*AF.step);

                for(int j=0; j<img.cols; j++){

                    GG::Vec B(bptr,j);
                    GG::Vec O(ptr,j);


                    GG::Vec BO=O-B;
                    GG::Vec BOu=BO;
                    BOu.unit();

                    double BOn2 = BO.norm2();
                    double BOn = sqrt(BOn2);


                    if (BOn==0){
                        //O==B -> Background
                        argF = K;
                        alpha = 0;
                    } else {
                        if(BOn>=(double)backgroundDistance){
                            //TODO this th must be the same as the one to remove clusters ?
                            //Foreground
                            argF=O;
                            alpha=1.0;
                        } else
                        {

                            //For each cluster
                            double minCost = DBL_MAX;
                            int nbC=0;

                            std::list<GG::Vec>::iterator it;
                            for(it=clusters.begin(); it!=clusters.end(); it++){

                                GG::Vec C = *it;



                                GG::Vec BC=C-B;
                                GG::Vec F=BOu * GG::Vec::dot(BOu,BC) + B;

                                GG::Vec CF=F-C;
                                double costCF = CF.norm2();

                                GG::Vec OF=F-O;
                                double costOF = OF.norm2();

                                GG::Vec BF=F-B;
                                double costBF = BF.norm2();

                                if(costBF<(double)(backgroundDistance*backgroundDistance)){
                                    //Wrong can't be solution !! as we defined F>distance
                                    continue;
                                }

                                double cost = costCF; //costOF*costCF;

                                if(i==244 && j==350){
                                    std::cout << "i=" << i << " j=" << j << "\n";
                                    std::cout << "O=" << O << " C=" << C << " B=" << B << "\n";
                                    std::cout << "F=" << F << " cost(CF)=" << sqrt(costCF) << " cost(OF)=" << sqrt(costOF) <<  " cost(BO)=" << BOn << " cost(BF)=" << sqrt(costBF) << "\n";
                                    std::cout << "cost=" << sqrt(cost) << "\n";
                                }

                                if(cost<minCost){
                                    minCost=cost;
                                    argF=F;
                                    argAlpha=0;
                                }

                                nbC++;

                            }//end for each cluster

                            //

                            //Clamp argF to colorspace
                            argF.x=(argF.x<0)?0:argF.x;
                            argF.x=(argF.x>100)?100:argF.x;
                            argF.y=(argF.y<-127)?-127:argF.y;
                            argF.y=(argF.y>127)?127:argF.y;
                            argF.z=(argF.z<-127)?-127:argF.z;
                            argF.z=(argF.z>127)?127:argF.z;
                            //

                            if(nbC==0){ // || minCost>50.0*50.0){
//                                argF=Y;

                                //Projects to bounadry
                                double d;
                                GG::Vec P0,n, X, bestX;

                                //Face A
                                P0=GG::Vec(50,-128,0);
                                n=GG::Vec(0,1,0);
                                X=GG::Vec::interRayPlane(B, BOu, P0, n, d);
                                if (X.x>=0 && X.x<=100 && X.y>-128 && X.y<128 && X.z>-128 && X.z<128 && d>0) {
                                    bestX=X;
                                }
                                //Face B
                                P0=GG::Vec(50,0,127);
                                n=GG::Vec(0,0,-1);
                                X=GG::Vec::interRayPlane(B, BOu, P0, n, d);
                                if (X.x>=0 && X.x<=100 && X.y>-128 && X.y<128 && X.z>-128 && X.z<128 && d>0){
                                    bestX=X;
                                }
                                //Face C
                                P0=GG::Vec(50,128,0);
                                n=GG::Vec(0,-1,0);
                                X=GG::Vec::interRayPlane(B, BOu, P0, n, d);
                                if (X.x>=0 && X.x<=100 && X.y>-128 && X.y<128 && X.z>-128 && X.z<128 && d>0){
                                    bestX=X;
                                }
                                //Face D
                                P0=GG::Vec(50,0,-128);
                                n=GG::Vec(0,0,1);
                                X=GG::Vec::interRayPlane(B, BOu, P0, n, d);
                                if (X.x>=0 && X.x<=100 && X.y>-128 && X.y<128 && X.z>-128 && X.z<128 && d>0){
                                    bestX=X;
                                }
                                //Face E
                                P0=GG::Vec(100,0,0);
                                n=GG::Vec(-1,0,0);
                                X=GG::Vec::interRayPlane(B, BOu, P0, n, d);
                                if (X.x>=0 && X.x<=100 && X.y>-128 && X.y<128 && X.z>-128 && X.z<128 && d>0) {
                                    bestX=X;
                                }
                                //Face F
                                P0=GG::Vec(0,0,0);
                                n=GG::Vec(1,0,0);
                                X=GG::Vec::interRayPlane(B, BOu, P0, n, d);
                                if (X.x>=0 && X.x<=100 && X.y>-128 && X.y<128 && X.z>-128 && X.z<128 && d>0){
                                    bestX=X;
                                }
                                //

                                argF=bestX;

                            }//end projection

//                            if(minCost>50.0*50.0){
//                                argF=Y;
//                            }

                            //

                        }

                    }//end else pure background

                    GG::Vec BF=argF-B;
                    argAlpha=BOn2/BF.norm2();

                    pA[j] = argAlpha;

                    pF[j*3]=argF.x;
                    pF[j*3+1] = argF.y;
                    pF[j*3+2] = argF.z;

                    pAF[j*3]=argAlpha*argF.x;
                    pAF[j*3+1] = argAlpha*argF.y;
                    pAF[j*3+2] = argAlpha*argF.z;

                }//end for j
            }//end for i

            return A;

        }//end alf 2



        static double findF1(GG::Vec &B, GG::Vec &C, GG::Vec &BO, GG::Vec &BC,
                            GG::Vec &F){


            double alpha = GG::Vec::dot(BO, BC)/BC.norm2();

            GG::Vec r=(BC*alpha) - BO;
            double rho=r.norm2();

            F=BO*(BC.norm2()/GG::Vec::dot(BO,BC))+B;

            return rho;
        }

        static double findF2(GG::Vec &B, GG::Vec &C, GG::Vec &BO, GG::Vec &BC,
                             GG::Vec &F){


            F=BO*(GG::Vec::dot(BO,BC)/BO.norm2())+B;
            GG::Vec CF=F-C;

            return CF.norm2();
        }

        static double findF3(GG::Vec &B, GG::Vec &C, GG::Vec &BO, GG::Vec &BC,
                             GG::Vec &F){


            F=(BO * (BC.norm()/BO.norm()))+B;
            double theta=acos(GG::Vec::dot(BC,BO)/(BC.norm()*BO.norm()));

            theta=(theta<0)?-theta:theta;

            return theta;
        }

        static void adjustF(double BOn, double OFn, GG::Vec &B, GG::Vec &O, GG::Vec &F){
            double w=1.0-pow(OFn/100.0,2.0);
            w=(w<0)?0:w;
            GG::Vec F2=(w*O)+(1.0-w)*F;
            F=F2;
            //Adjust alpha
            GG::Vec BF=F-B;
            F.alpha = BOn/BF.norm();
        }


        static cv::Mat goAlf0(cv::Mat &img, cv::Mat &back, std::list<GG::Vec> &clusters,
                              double backgroundDistance, double distTh,
                              cv::Mat &Fmap, cv::Mat &AF, int method){//),
//                              cv::Mat &GTF){

            cv::Mat A=cv::Mat::zeros(img.rows, img.cols, CV_32FC1);

            double alpha;
            GG::Vec K(0,0,0,0);
            GG::Vec Y(98.0,-16.0,93.0,0);
            GG::Vec F;

            for(int i=0; i<img.rows; i++){

                float *ptr = (float*) (img.data + i*img.step);
                float *bptr = (float*) (back.data + i*back.step);
                float *pA = (float*) (A.data+i*A.step);
                float *pF = (float*) (Fmap.data+i*Fmap.step);
                float *pAF = (float*) (AF.data+i*AF.step);
//                float *pGTF = (float*) (GTF.data+i*GTF.step);

                for(int j=0; j<img.cols; j++){

                    GG::Vec B(bptr,j);
                    GG::Vec O(ptr,j);

                    GG::Vec BO=O-B;
                    GG::Vec OB=B-O;
                    GG::Vec BOu=BO;
                    BOu.unit();

                    double BOn2 = BO.norm2();
                    double BOn = sqrt(BOn2);


                    if (BOn==0){
                        //O==B -> Background
                        F = K;
                        alpha = 0;
                    } else {
                        if(BOn>=(double)backgroundDistance){
                            //Foreground
                            F=O;
                            alpha=1.0;
                        } else
                        {

                            //For each cluster
                            std::set<GG::Vec, GG::Vec::vComp> candidates;
                            std::list<GG::Vec>::iterator it;
                            for(it=clusters.begin(); it!=clusters.end(); it++){

                                GG::Vec C = *it;
                                GG::Vec BC=C-B;
                                GG::Vec BF, OF;

                                double dist=DBL_MAX;

                                double direction = GG::Vec::dot(BO,BC);

//                                if(direction>=0){
                                switch (method) {
                                    case 1:
                                        dist = findF1(B, C, BO, BC, F);
                                        break;
                                    case 2:
                                        dist = findF2(B, C, BO, BC, F);
                                        break;
                                    case 3:
                                        dist = findF3(B, C, BO, BC, F);
                                        break;
                                    default:
                                        break;
                                }

                                    BF=F-B;
                                    OF=F-O;
//                                } else {
//                                    //Swap B<->O
//                                    GG::Vec OC=C-O;
//                                    dist = findF3(O, C, OB, OC, F);
//                                    BF=F-O;
//                                    OF=F-B;
//                                }

                                if(direction>0){
                                    F.alpha = BOn/BF.norm();
                                } else {
                                    F.alpha = 0;
                                }

                                F.u = dist;
                                F.v = OF.norm();

//                                if(i==370 && j==340){
//                                    std::cout << "i=" << i << " j=" << j << " \n";
//                                    std::cout << "\tO=" << O << " B=" << B << " C=" << C << " F=" << F << "\n";
//                                    std::cout << "\tdist=" << sqrt(dist) << " a=" << F.alpha << " OF=" << F.v << "\n";
//                                    if(dist<distTh*distTh){
//                                        std::cout << "\t\tACCEPT--\n";
//                                    } else {
//                                        std::cout << "\t\tREJECT--\n";
//                                    }
//                                }

                                if(dist<distTh*distTh){
                                    candidates.insert(F);
                                }



                            }//end for each cluster

                            if(candidates.size()>0){
                                //Get the best candidate !!
                                std::set<GG::Vec>::iterator itc=candidates.begin();
                                F=*itc;
                                alpha=F.alpha;
//                                 if(i==370 && j==340){
//                                     std::cout << "BEST dist=" << sqrt(F.u) << " BEST OF=" << F.v << "\n\n";
//                                 }
                            } else {
                                //No candidates, assume background
//                                F=Y;
//                                alpha=0.5;
                                F=Y;
                                alpha=0;
                            }


                        }

                    }//end if


                    //If ||BF||<backgroundDistance, push F to boundary
                    GG::Vec BF=F-B;
                    if(BF.norm2()<backgroundDistance*backgroundDistance){
                        GG::Vec BOunit = BO*(1.0/BOn);
                        BF=BOunit*backgroundDistance;
                        F=BF+B;
                        alpha=BOn/BF.norm();
                    }


                    //Adjust F in respect with OF
                    //TODO is that useless or not ?
//                    adjustF(BOn, F.v, B, O, F);
//                    alpha = F.alpha;
//                    //Make sure ||BF||>th
//                    GG::Vec BF=F-B;
//                    if(BF.norm2()<backgroundDistance*backgroundDistance){
//                        std::cout << "\tBAD \t" << BF.norm() << "<" << backgroundDistance <<"\n";
//                    }
//                    if(i==300 && j==440){
//                        std::cout << "Ajusted to:\n";
//                        std::cout << "\tF=" << F << "\n";
//                        std::cout << "\ta=" << F.alpha  << "\n";
//                    }

//                    if(F.v<15.0){
//                        std::cout << "i=" << i << " j=" << j << " O=" << O << " B=" << B << " F=" << F << " OF=" << F.v <<"\n";
//                    }

                    pA[j] = alpha;

                    pF[j*3]=F.x;
                    pF[j*3+1] = F.y;
                    pF[j*3+2] = F.z;

                    pAF[j*3] = alpha*F.x;
                    pAF[j*3+1] = alpha*F.y;
                    pAF[j*3+2] = alpha*F.z;

                }//end for j
            }//end for i

            return A;

        }//end alf 0

        static double eval(cv::Mat &AF, cv::Mat &GTAF){

            double sum=0;
            int cpt=0;

            for(int i=0; i<AF.rows; i++){

                float *pAF = (float*) (AF.data+i*AF.step);
                float *pGTAF = (float*) (GTAF.data+i*GTAF.step);

                for(int j=0; j<AF.cols; j++){

                    GG::Vec af(pAF,j);
                    GG::Vec gtaf(pGTAF,j);

                    GG::Vec d=gtaf-af;
                    sum+=d.norm();
                    cpt++;

                }
            }

            return sum/(double)cpt;

        }//end eval


    };// end class

} //end namespace

#endif /* defined(__alf__Alf__) */

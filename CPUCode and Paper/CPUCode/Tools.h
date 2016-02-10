//
//  Tools.h
//  alf
//
//  Created by Guillaume GALES on 02/05/2014.
//  Copyright (c) 2014 Guillaume Gales. All rights reserved.
//

#ifndef __alf__Tools__
#define __alf__Tools__

#include <iostream>
#include <opencv2/opencv.hpp>

namespace GG {
    
    /**
     * Provides tools for image processing (requires OpenCV)
     */
    class Tools {
        
    public:
        
        /**
         * RGB (8 or 16U) to Lab (32F)
         * 0<=L<=100 ; -127 <= a <= 127 ; -127 <= b <= 127
         */
        static cv::Mat RGBtoLab(cv::Mat &input){
            float coef;
            if(input.type()==CV_8U){
                coef=1.0f/255.0f;
            } else {
                if(input.type()==CV_16U){
                    coef=1.0f/65535.0f;
                }
            }
            cv::Mat lab;
            input.convertTo(lab, CV_32FC3);
            lab *= (1.0/255.0);
            cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);
            return lab;
        }
        
        /**
         * Lab (32U) to RGB (16U)
         */
        static cv::Mat Lab2RGB16(cv::Mat &input){
            cv::Mat rgb;
            cv::cvtColor(input, rgb, cv::COLOR_Lab2BGR);
            rgb *= 65535.0f;
            rgb.convertTo(rgb, CV_16UC3);
            return rgb;
        }
        
        /**
         * Lab (32U) to RGB (8U)
         */
        static cv::Mat Lab2RGB8(cv::Mat &input){
            cv::Mat rgb;
            cv::cvtColor(input, rgb, cv::COLOR_Lab2BGR);
            rgb *= 255.0f;
            rgb.convertTo(rgb, CV_8UC3);
            return rgb;
        }
        
        /**
         * Alpha to RGB (16U)
         */
        static cv::Mat Alpha2RGB16(cv::Mat &input){
            cv::Mat rgb=input;
            rgb *= 65535.0f;
            rgb.convertTo(rgb, CV_16UC1);
            return rgb;
        }
        
        /**
         * Lab to RGB8 (itemwise)
         */
        static void LAB2RGB(double L, double a, double b,
                            unsigned char &R, unsigned char &G, unsigned char &B)
        {
            float X, Y, Z, fX, fY, fZ;
            int RR, GG, BB;
            
            fY = pow((L + 16.0) / 116.0, 3.0);
            if (fY < 0.008856)
                fY = L / 903.3;
            Y = fY;
            
            if (fY > 0.008856)
                fY = powf(fY, 1.0/3.0);
            else
                fY = 7.787 * fY + 16.0/116.0;
            
            fX = a / 500.0 + fY;
            if (fX > 0.206893)
                X = powf(fX, 3.0);
            else
                X = (fX - 16.0/116.0) / 7.787;
            
            fZ = fY - b /200.0;
            if (fZ > 0.206893)
                Z = powf(fZ, 3.0);
            else
                Z = (fZ - 16.0/116.0) / 7.787;
            
            X *= (0.950456 * 255);
            Y *=             255;
            Z *= (1.088754 * 255);
            
            RR =  (int)(3.240479*X - 1.537150*Y - 0.498535*Z + 0.5);
            GG = (int)(-0.969256*X + 1.875992*Y + 0.041556*Z + 0.5);
            BB =  (int)(0.055648*X - 0.204043*Y + 1.057311*Z + 0.5);
            
            R = (unsigned char)(RR < 0 ? 0 : RR > 255 ? 255 : RR);
            G = (unsigned char)(GG < 0 ? 0 : GG > 255 ? 255 : GG);
            B = (unsigned char)(BB < 0 ? 0 : BB > 255 ? 255 : BB);

        }
        
    };// end class
    
} //end namespace

#endif /* defined(__alf__Tools__) */

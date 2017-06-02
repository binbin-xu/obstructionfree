//
//  main.cpp
//  opencv-test
//
//  Created by XU BINBIN on 4/4/17.
//  Copyright © 2017 XU BINBIN. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/optflow.hpp>


#include <stdlib.h>
#include <stdio.h>

#include "mrf.h"
#include"edgeflow.h"

using namespace cv;
using namespace std;

void colorFlow(Mat flow);
void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double, const Scalar& color);

int main(int argc, const char * argv[]) {

    //some parameters



    Mat im1,im2;
    Mat im1_grey, im2_grey;

    im1 =imread(argv[1]);
    im2 = imread(argv[2]);

    if( !im1.data )
    { return -1; }

    if( !im2.data )
    { return -1; }

    cvtColor(im1, im1_grey, COLOR_RGB2GRAY);
    cvtColor(im2, im2_grey, COLOR_RGB2GRAY);

    //downsample the images
    for (int i=0; i<3; i++){
        pyrDown( im1_grey, im1_grey );
        pyrDown( im2_grey, im2_grey );
    }


    Mat im1_edge, im2_edge;
    Canny(im1_grey, im1_edge, 10, 100,3,true);
    Canny(im2_grey, im2_edge, 10, 100,3,true);


    //replace edgeflow
    Ptr<DenseOpticalFlow> deepflow = optflow::createOptFlow_DeepFlow();
    Mat flow;
    deepflow->calc(im1_grey, im2_grey, flow);

    Mat edgeflow;
    flow.copyTo(edgeflow,im1_edge);
    colorFlow(edgeflow);

    //drawOptFlowMap(flow, im1_grey, 16, 1.5, CV_RGB(0, 255, 0));
    //imshow("flow", im1_grey);

    //cout<<flow.type()<<endl;


/*
    int patch=5; //patch size (2*patch+1)^2
    //remove edges near the image borders
    for (int y=0; y<im1_edge.rows;y++){
        for (int x=0; x<im1_edge.cols; x++){
            if ( y < patch || x < patch || x >= (im1_edge.cols-patch) || y >= (im1_edge.rows-patch) ){
                im1_edge.at<uchar>(y,x)=0;
                im2_edge.at<uchar>(y,x)=0;
            }
        }
    }

    //#label=#edge in image2
    Mat label_Locations;
    findNonZero(im2_edge, label_Locations);
    int nL = label_Locations.total();

    // data-term
    MRF::CostVal *cData = NULL;
	computeCost(im1_grey, im2_grey, im1_edge, im2_edge, cData, patch, nL);
	DataCost *dcost = new DataCost(cData);


    SmoothnessCost *scost;
	MRF::CostVal *hCue = NULL, *vCue = NULL;
	if (gradThresh > 0) {
	    computeCues(im1, hCue, vCue, gradThresh, gradPenalty);
	    scost = new SmoothnessCost(smoothexp, smoothmax, lambda, hCue, vCue);
	} else {
	    scost = new SmoothnessCost(smoothexp, smoothmax, lambda);
	}
	EnergyFunction *energy = new EnergyFunction(dcost, scost);

	*/

    //namedWindow("canny",WINDOW_AUTOSIZE);
    //imshow("canny", im2_edge);

    cv::waitKey(0);

}


void colorFlow(Mat flow)
{
    //extraxt x and y channels
    cv::Mat xy[2]; //X,Y
    cv::split(flow, xy);

    //calculate angle and magnitude
    cv::Mat magnitude, angle;
    cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);

    //translate magnitude to range [0;1]
    double mag_max;
    cv::minMaxLoc(magnitude, 0, &mag_max);
    magnitude.convertTo(magnitude, -1, 1.0 / mag_max);

    //build hsv image
    cv::Mat _hsv[3], hsv;
    _hsv[0] = angle;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magnitude;
    cv::merge(_hsv, 3, hsv);

    //convert to BGR and show
    cv::Mat bgr;//CV_32FC3 matrix
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::imshow("optical flow", bgr);
    //imwrite("c://resultOfOF.jpg", bgr);
    //cv::waitKey(0);
}

void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
                    double, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

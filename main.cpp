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

void colorFlow(Mat flow, string figName);
void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double, const Scalar& color);
Mat indexToMask(Mat indexMat, int rows, int cols);

//use Homography to filter outliers in the flow
Mat flowHomography(Mat edges, Mat flow, int ransacThre);

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
    colorFlow(edgeflow,"edge_flow");


    //flow=>points
    Mat backH, mask_backH;
    int back_ransacThre=1;
    vector<Point> edge_Locations1;
    findNonZero(im1_edge, edge_Locations1);

//    backH = flowHomography(im1_edge, edgeflow, back_ransacThre, mask_backH);


//
//    vector<Point> obj_edgeflow;
//
//    for(size_t i = 0; i<edge_Locations1.size();i++){
//        int src_x=edge_Locations1[i].x;
//        int src_y=edge_Locations1[i].y;
//        Point2f f = flow.at<Point2f>(src_y, src_x);
//        obj_edgeflow.push_back(Point2i(src_x + f.x, src_y + f.y));
//    }
//
//    backH = findHomography( edge_Locations1, obj_edgeflow, RANSAC, back_ransacThre, mask_backH );

    //edges fit background homography
//    Mat back_edges, back_edgeLocations;
//    Mat(edge_Locations1).copyTo(back_edgeLocations,mask_backH);


    //convert index matrix to mask matrix
    //back_edges=indexToMask(back_edgeLocations, im1_edge.rows, im1_edge.cols);

    Mat back_edges=flowHomography(im1_edge, edgeflow, back_ransacThre);
    imshow("back_edges", back_edges);

    Mat back_flow;
    edgeflow.copyTo(back_flow,back_edges);
    colorFlow(back_flow, "back_flow");

    //rest edges and flows
    Mat rest_edges=im1_edge-back_edges;
    imshow("rest_edges", rest_edges);
    Mat rest_flow=edgeflow-back_flow;
    colorFlow(rest_flow, "rest_flow");

    //align resting flows to another homograghy
    Mat foreH, mask_foreH;
    int fore_ransacThre=1;

    vector<Point> restedge_Locations;
    findNonZero(rest_edges, restedge_Locations);

    //foreH = flowHomography(rest_edges, rest_flow, fore_ransacThre, mask_foreH);

    //Mat fore_edges, fore_edgeLocations;
    //Mat(restedge_Locations).copyTo(fore_edgeLocations,mask_foreH);


    //convert index matrix to mask matrix
    //fore_edges=indexToMask(fore_edgeLocations, im1_edge.rows, im1_edge.cols);
    Mat fore_edges=flowHomography(rest_edges, rest_flow, fore_ransacThre);
    imshow("fore_edges", fore_edges);

    Mat fore_flow;
    rest_flow.copyTo(fore_flow,fore_edges);
    colorFlow(fore_flow, "fore_flow");



//    imshow("rest_edges", rest_edges);




    //cout<<back_edgeLocations.cols<<endl<<back_edgeLocations.rows<<endl;
    //cout<< edgeflow.size()<<endl<< mask_backH.total()<<endl;
    //cout<<back_edgeLocations.type()<<endl;
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


void colorFlow(Mat flow, string figName="optical flow")
{
    //extraxt x and y channels
    Mat xy[2]; //X,Y
    split(flow, xy);

    //calculate angle and magnitude
    Mat magnitude, angle;
    cartToPolar(xy[0], xy[1], magnitude, angle, true);

    //translate magnitude to range [0;1]
    double mag_max;
    minMaxLoc(magnitude, 0, &mag_max);
    magnitude.convertTo(magnitude, -1, 1.0 / mag_max);

    //build hsv image
    Mat _hsv[3], hsv;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magnitude;
    merge(_hsv, 3, hsv);

    //convert to BGR and show
    Mat bgr;//CV_32FC3 matrix
    cvtColor(hsv, bgr, COLOR_HSV2BGR);
    imshow(figName, bgr);
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

    Mat indexToMask(Mat indexMat, int rows, int cols){
        Mat maskMat=Mat::zeros(rows, cols, CV_8UC1);
        for (int i = 0; i < indexMat.cols; i++ ) {
        for (int j = 0; j < indexMat.rows; j++) {
                Vec2i mask_loca = indexMat.at<Vec2i>(j, i);
                if (mask_loca[0] !=0 && mask_loca[1] !=0) {
                    maskMat.at<uchar>(Point(mask_loca)) = 255;}
        }}
    return  maskMat;
    }

    Mat flowHomography(Mat edges, Mat flow, int ransacThre){
        Mat inlierMask, inlier_edges, inilier_edgeLocations;
        vector<Point> edge_Locations1;

        findNonZero(edges, edge_Locations1);

        vector<Point> obj_edgeflow;

        for(size_t i = 0; i<edge_Locations1.size();i++){
            int src_x=edge_Locations1[i].x;
            int src_y=edge_Locations1[i].y;
            Point2f f = flow.at<Point2f>(src_y, src_x);
            obj_edgeflow.push_back(Point2i(src_x + f.x, src_y + f.y));
        }

        Mat Homography = findHomography( edge_Locations1, obj_edgeflow, RANSAC, ransacThre, inlierMask);

        Mat(edge_Locations1).copyTo(inilier_edgeLocations,inlierMask);

        //convert index matrix to mask matrix
        inlier_edges=indexToMask(inilier_edgeLocations, edges.rows, edges.cols);

        return inlier_edges;
    }


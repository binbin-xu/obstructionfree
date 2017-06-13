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

#include <opencv2/ximgproc/sparse_match_interpolator.hpp>

//#include <gsl/gsl_math.h>
//#include <gsl/gsl_interp2d.h>
//#include <gsl/gsl_spline2d.h>

//#include "interpolator.h"

#include <stdlib.h>
#include <stdio.h>

//#include "mrf.h"
#include"edgeflow.h"

using namespace cv;
using namespace std;

void colorFlow(Mat flow, string figName);
void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double, const Scalar& color);
Mat indexToMask(Mat indexMat, int rows, int cols);

//use Homography to filter outliers in the flow
Mat flowHomography(Mat edges, Mat flow, int ransacThre);

//interpolate from sparse edgeflow to dense optical flow
Mat sparse_int_dense(Mat im1, Mat im2, Mat im1_edges, Mat sparseFlow);

Mat imgWarpFlow(Mat im1, Mat flow);

int main(int argc, const char * argv[]) {

    //some parameters
    int back_ransacThre=1;
    int fore_ransacThre=1;
    int pyramid_level=3;

    vector<Mat> back_flowfields;
    vector<Mat> fore_flowfields;
    vector<Mat> warpedToReference;
    Mat alpha_map;
    Mat background;
    Mat foregrond;

    ////////////////////input image sequences//////////////////////
    vector<Mat> video_input;
    Mat referFrame;
    Mat currentFrame;

    VideoCapture capture(argv[2]);
    int frameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
    int reference_number=(frameNumber-1)/2;

    //namedWindow("inputVideo", WINDOW_AUTOSIZE);
    //cout<<frameNumber<<endl<<reference_number<<endl;
    for(int frame_i=0; frame_i < frameNumber; frame_i++)
    {
       capture>>currentFrame;
       if(frame_i==reference_number){referFrame=currentFrame.clone();}
       video_input.push_back(currentFrame.clone());
       //imshow("inputVideo", currentFrame);
       //waitKey(10);
    }
    //imshow("referFrame", referFrame);

    /////////////////initialization->motion fields for back/foreground layers///////////
    for (int frame_i=0; frame_i<frameNumber; frame_i++){
        Mat im1,im2;//reference frame, other frame
        Mat im1_grey, im2_grey;
        Mat im1_edge, im2_edge;
        Mat flow;
        Mat edgeflow;  //extracted edgeflow

        //Mat backH, mask_backH;
        Mat back_edges, rest_edges, fore_edges; //edges aligned to the back layer using homography, remaining layer, foreground layers
        Mat back_flow, rest_flow, fore_flow;
        //Mat foreH, mask_foreH;
        Mat back_denseFlow, fore_denseFlow;

        if (frame_i!=reference_number){
            //int frame_i=1;
            im1 = referFrame.clone();
            im2 = video_input[frame_i].clone();

            cvtColor(im1, im1_grey, COLOR_RGB2GRAY);
            cvtColor(im2, im2_grey, COLOR_RGB2GRAY);

            //////////downsample the images
            for (int i=0; i<pyramid_level; i++){
                pyrDown( im1_grey, im1_grey );
                pyrDown( im2_grey, im2_grey );
            }


            Canny(im1_grey, im1_edge, 10, 100,3,true);
            Canny(im2_grey, im2_edge, 10, 100,3,true);

            ///////////////replace edgeflow
            Ptr<DenseOpticalFlow> deepflow = optflow::createOptFlow_DeepFlow();
            deepflow->calc(im1_grey, im2_grey, flow);
           // colorFlow(flow,"optical_flow");
            flow.copyTo(edgeflow,im1_edge);
            //colorFlow(edgeflow,"edge_flow");

  ////////flow=>points using homography-ransac filtering, and then extract flow on the filtered edges
            back_edges=flowHomography(im1_edge, edgeflow, back_ransacThre);
           // imshow("back_edges", back_edges);
            edgeflow.copyTo(back_flow,back_edges);
           // colorFlow(back_flow, "back_flow");
            //////////rest edges and flows
            rest_edges=im1_edge-back_edges;
            //imshow("rest_edges", rest_edges);
            rest_flow=edgeflow-back_flow;
           // colorFlow(rest_flow, "rest_flow");

            ////////////align resting flows to another homograghy
            fore_edges=flowHomography(rest_edges, rest_flow, fore_ransacThre);
           // imshow("fore_edges", fore_edges);
            rest_flow.copyTo(fore_flow,fore_edges);
            //colorFlow(fore_flow, "fore_flow");

    ///////////////////interpolation from sparse edgeFlow to denseFlow/////////////////////
            back_denseFlow=sparse_int_dense(im1_grey, im2_grey, back_edges, back_flow);
            fore_denseFlow=sparse_int_dense(im1_grey, im2_grey, fore_edges, fore_flow);
            //cout<<back_denseFlow.type()<<endl;
            back_flowfields.push_back(back_denseFlow.clone());
            fore_flowfields.push_back(fore_denseFlow.clone());
            //colorFlow(back_denseFlow,"inter_back_denseflow");
            //colorFlow(fore_denseFlow,"inter_fore_denseflow");
//
    ////////////warping images to the reference frame///////////////////
            Mat warpedFrame=imgWarpFlow(im2_grey, back_denseFlow);
            warpedToReference.push_back(warpedFrame.clone());
            //imshow("warped image",warpedFrame);
        }
        else{
            Mat refer_grey;
            cvtColor(referFrame, refer_grey, COLOR_RGB2GRAY);
            for (int i=0; i<pyramid_level; i++){
                pyrDown( refer_grey, refer_grey );
            }
            warpedToReference.push_back(refer_grey.clone());
            back_flowfields.push_back(Mat::zeros(referFrame.rows,referFrame.cols,CV_32FC2));
            fore_flowfields.push_back(Mat::zeros(referFrame.rows,referFrame.cols,CV_32FC2));
        }
        }
        ////////////show warped image frames/////////////
        for (int frame_i=0; frame_i<frameNumber; frame_i++){
            char windowName[10];
            sprintf(windowName, "warped %d", frame_i);
            imshow(windowName,warpedToReference[frame_i]);
        }


           //////////////////////////Initialization/////////////////
           /////// opaque occlusion/////////////
    Mat sum=Mat::zeros(warpedToReference[reference_number].rows,warpedToReference[reference_number].cols,CV_32F);
    Mat temp,background_temp;
    for (int frame_i=0; frame_i<frameNumber; frame_i++){
        warpedToReference[frame_i].convertTo(temp,CV_32F);
        sum+=temp;
    }
    background_temp=sum/frameNumber;
    background_temp.convertTo(background,CV_8UC1);
    imshow("opaque initial background", background);

    warpedToReference[reference_number].convertTo(temp,CV_32F);
    //cout<<temp.type()<<endl;
    Mat difference;
    difference=abs(background-warpedToReference[reference_number]);
    difference.convertTo(difference,CV_32F);
    //cout<<difference<<endl;

    //cout<<difference.type()<<endl;
    threshold(difference, alpha_map,25.5,1,THRESH_BINARY_INV);
    imshow("alpha map",alpha_map);
    //cout<<alpha_map<<endl;


//            ////////  reflection pane///////////////////
//    background=warpedToReference[reference_number];
//    for (int frame_i=0; frame_i<frameNumber; frame_i++){
//        background=min(background,warpedToReference[frame_i]);
//    }
//    imshow("reflection initial background", background);
//
//





/////////////////////MRF////////////////////////////////////////////////
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


    //interpolation



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

    Mat sparse_int_dense(Mat im1, Mat im2, Mat im1_edges, Mat sparseFlow){
        Mat denseFlow;
        vector<Point2f> sparseFrom;
        vector<Point2f> sparseTo;

        vector<Point> edge_Location;
        findNonZero(im1_edges, edge_Location);
        for(size_t i = 0; i<edge_Location.size();i++){
            float src_x=edge_Location[i].x;
            float src_y=edge_Location[i].y;
            sparseFrom.push_back(Point2f(src_x, src_y));
            Point2f f = sparseFlow.at<Point2f>(src_y, src_x);
            sparseTo.push_back(Point2f(src_x + f.x, src_y + f.y));
        }

        Ptr<cv::ximgproc::SparseMatchInterpolator> epicInterpolation=ximgproc::createEdgeAwareInterpolator();
        epicInterpolation->interpolate(im1,sparseFrom,im2,sparseTo,denseFlow);
        return denseFlow;
    }

    //flow=flow->cal(im1,im2), so warp im2 to back
    Mat imgWarpFlow(Mat im1, Mat flow){
        Mat flowmap_x(flow.size(), CV_32FC1);
        Mat flowmap_y(flow.size(), CV_32FC1);
        for (int j = 0; j < flowmap_x.rows; j++){
            for (int i = 0; i < flowmap_x.cols; ++i){
                Point2f f = flow.at<Point2f>(j, i);
                flowmap_y.at<float>(j, i) = float(j + f.y);
                flowmap_x.at<float>(j, i) = float(i + f.x);
                }}
        Mat warpedFrame;
        remap(im1, warpedFrame, flowmap_x,flowmap_y ,INTER_CUBIC,BORDER_CONSTANT,255);
        return warpedFrame;
    }


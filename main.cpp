//
//  main.cpp
//  opencv-test
//
//  Created by XU BINBIN on 4/4/17.
//  Copyright © 2017 XU BINBIN. All rights reserved.
//

#include <iostream>
#include <opencv2/core.hpp>
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


int pyramid_level=3;
size_t frameNumber;
size_t reference_number;

void colorFlow(Mat flow, string figName);
void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double, const Scalar& color);
Mat indexToMask(Mat indexMat, int rows, int cols);

//use Homography to filter outliers in the flow
Mat flowHomography(Mat edges, Mat flow, int ransacThre);

//interpolate from sparse edgeflow to dense optical flow
Mat sparse_int_dense(Mat im1, Mat im2, Mat im1_edges, Mat sparseFlow);

Mat imgWarpFlow(Mat im1, Mat flow);

//add flow src_flow + add_flow=>obj_flow
Mat addFlow(Mat src_flow, Mat add_flow);

void initila_motion_decompose(Mat im1, Mat im2, Mat& back_denseFlow, Mat& fore_denseFlow, int back_ransacThre, int fore_ransacThre);
//motion fields initialization
//direct matching to the reference frame
void motion_initiliazor_direct(const vector<Mat>& video_input, vector<Mat>& back_flowfields, vector<Mat>& fore_flowfields, vector<Mat>& warpedToReference);
//matching between neighbouring frames and warping to the reference frame
void motion_initiliazor_iterative(const vector<Mat>& video_input, vector<Mat>& back_flowfields, vector<Mat>& fore_flowfields, vector<Mat>& warpedToReference);
//irls motion decomposition
void mot_decom_irls(const vector<Mat>& input_sequence, Mat& backgd_comp, Mat& obstruc_comp, Mat& alpha_map, vector<Mat> back_flowfields, vector<Mat> fore_flowfields,  int nOuterFPIterations);

int main(int argc, const char * argv[]) {

    //some parameters


    vector<Mat> back_flowfields;
    vector<Mat> fore_flowfields;
    vector<Mat> warpedToReference;
    Mat alpha_map;
    Mat background;
    Mat foregrond;

    ////////////////////input image sequences//////////////////////
    vector<Mat> video_input;
    vector<Mat> video_coarseLeve;
    Mat referFrame;
    Mat currentFrame;

    VideoCapture capture(argv[2]);
    frameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
    reference_number=(frameNumber-1)/2;

    //namedWindow("inputVideo", WINDOW_AUTOSIZE);
    //cout<<frameNumber<<endl<<reference_number<<endl;
    for(size_t frame_i=0; frame_i < frameNumber; frame_i++)
    {
       capture>>currentFrame;
       if(frame_i==reference_number){referFrame=currentFrame.clone();}
       video_input.push_back(currentFrame.clone());
       //imshow("inputVideo", currentFrame);
       //waitKey(10);
    }
    //imshow("referFrame", referFrame);

    /////////construct image pyramids//////
    for (size_t frame_i=0; frame_i<frameNumber; frame_i++){
        Mat temp, temp_gray;
        temp=video_input[frame_i].clone();
        cvtColor(temp, temp_gray, COLOR_RGB2GRAY);
        for (int i=0; i<pyramid_level; i++){
            pyrDown( temp_gray, temp_gray );
        }
        video_coarseLeve.push_back(temp_gray.clone());
    }

    /////////////////initialization->motion fields for back/foreground layers///////////
    motion_initiliazor_direct(video_coarseLeve, back_flowfields, fore_flowfields, warpedToReference);
    //motion_initiliazor_iterative(video_coarseLeve, back_flowfields, fore_flowfields, warpedToReference);
        ////////////show warped image frames/////////////
        for (size_t frame_i=0; frame_i<frameNumber; frame_i++){
            char windowName[10];
            sprintf(windowName, "warped %zu", frame_i);
            imshow(windowName,warpedToReference[frame_i]);
            //char flowwindow[10];
            //sprintf(flowwindow, "flow %d", frame_i);
            //colorFlow(back_flowfields[frame_i], flowwindow);
        }


           //////////////////////////Initialization/////////////////

           /////// opaque occlusion/////////////
    Mat sum=Mat::zeros(warpedToReference[reference_number].rows,warpedToReference[reference_number].cols,CV_32F);
    Mat temp,background_temp;
    for (size_t frame_i=0; frame_i<frameNumber; frame_i++){
        warpedToReference[frame_i].convertTo(temp,CV_32F);
        sum+=temp;
    }

    background_temp=sum/frameNumber;
    background_temp.convertTo(background,CV_8UC1);
    imshow("opaque initial background", background);

    warpedToReference[reference_number].convertTo(temp,CV_32F);
    Mat difference;
    difference=abs(background-warpedToReference[reference_number]);
    threshold(difference, alpha_map,25.5,255,THRESH_BINARY_INV);
    imshow("alpha map",alpha_map);
    //cout<<alpha_map<<endl;

    foregrond=warpedToReference[reference_number]-background;
    imshow("foreground",foregrond);


//            ////////  reflection pane///////////////////
//    background=warpedToReference[reference_number];
//    for (size_t frame_i=0; frame_i<frameNumber; frame_i++){
//        background=min(background,warpedToReference[frame_i]);
//    }
//    imshow("reflection initial background", background);





////////////////////IRLS decomposition/////////////////


////////////////////IRLS motion estimation/////////////////

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

  void initila_motion_decompose(Mat im1, Mat im2, Mat& back_denseFlow, Mat& fore_denseFlow, int back_ransacThre=1, int fore_ransacThre=1){
        if (im1.channels()!= 1)
            cvtColor(im1, im1, COLOR_RGB2GRAY);
        if (im2.channels()!= 1)
            cvtColor(im2, im2, COLOR_RGB2GRAY);

        Mat im1_edge, im2_edge;
        Mat flow;
        Mat edgeflow;  //extracted edgeflow

        //Mat backH, mask_backH;
        Mat back_edges, rest_edges, fore_edges; //edges aligned to the back layer using homography, remaining layer, foreground layers
        Mat back_flow, rest_flow, fore_flow;


        Canny(im1, im1_edge, 10, 100,3,true);
        Canny(im2, im2_edge, 10, 100,3,true);

        ///////////////replace edgeflow
        Ptr<DenseOpticalFlow> deepflow = optflow::createOptFlow_DeepFlow();
        deepflow->calc(im1, im2, flow);
        //colorFlow(flow,"optical_flow");
        flow.copyTo(edgeflow, im1_edge);
        //colorFlow(edgeflow,"edge_flow");

    ////////flow=>points using homography-ransac filtering, and then extract flow on the filtered edges
        back_edges=flowHomography(im1_edge, edgeflow, back_ransacThre);
        //imshow("back_edges", back_edges);
        edgeflow.copyTo(back_flow,back_edges);
        //colorFlow(back_flow, "back_flow");
        //////////rest edges and flows
        rest_edges=im1_edge-back_edges;
        //imshow("rest_edges", rest_edges);
        rest_flow=edgeflow-back_flow;
       // colorFlow(rest_flow, "rest_flow");

        ////////////align resting flows to another homograghy
        fore_edges=flowHomography(rest_edges, rest_flow, fore_ransacThre);
        //imshow("fore_edges", fore_edges);
        rest_flow.copyTo(fore_flow,fore_edges);
        //colorFlow(fore_flow, "fore_flow");

    ///////////////////interpolation from sparse edgeFlow to denseFlow/////////////////////
        back_denseFlow=sparse_int_dense(im1, im2, back_edges, back_flow);
        fore_denseFlow=sparse_int_dense(im1, im2, fore_edges, fore_flow);
        //colorFlow(back_denseFlow,"inter_back_denseflow");
        //colorFlow(fore_denseFlow,"inter_fore_denseflow");
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

    //add flow src_flow + add_flow=>obj_flow
    Mat addFlow(Mat src_flow, Mat add_flow){
        Mat obj_flow=src_flow.clone();
        int src_x, src_y;
        float obj_y, obj_x;

        for (int j = 0; j < src_flow.rows; j++){
            for (int i = 0; i < src_flow.cols; ++i){
                Point2f src_f = src_flow.at<Point2f>(j, i);
                src_y = int(j + src_f.y);
                if (src_y >= src_flow.rows){src_y=src_flow.rows-1;}
                src_x = int(i + src_f.x);
                if (src_x >= src_flow.cols){src_x=src_flow.cols-1;}

                Point2f add_f = add_flow.at<Point2f>(src_y, src_x);
                obj_y = float(src_y + add_f.y);
                if (obj_y >= src_flow.rows){obj_y = src_flow.rows-1;}
                obj_x = float(src_x + add_f.x);
                if (obj_x >= src_flow.cols){obj_x = src_flow.cols-1;}
                obj_flow.at<Point2f>(j, i) = Point2f(obj_x - i, obj_y - j);
        }}
        return obj_flow;
    }

void motion_initiliazor_direct(const vector<Mat>& video_input, vector<Mat>& back_flowfields, vector<Mat>& fore_flowfields, vector<Mat>& warpedToReference){
    int back_ransacThre=1;
    int fore_ransacThre=1;

    for (size_t frame_i=0; frame_i<frameNumber; frame_i++){
        Mat im1,im2;//reference frame, other frame

        //Mat foreH, mask_foreH;
        Mat back_denseFlow, fore_denseFlow;

        if (frame_i!=reference_number){
            //int frame_i=1;
            im1 = video_input[reference_number].clone();
            im2 = video_input[frame_i].clone();

            //decompose motion fields into fore/background
            initila_motion_decompose(im1, im2, back_denseFlow, fore_denseFlow, back_ransacThre, fore_ransacThre);

            //cout<<back_denseFlow.type()<<endl;
            back_flowfields.push_back(back_denseFlow.clone());
            fore_flowfields.push_back(fore_denseFlow.clone());
            //colorFlow(back_denseFlow,"inter_back_denseflow");
            //colorFlow(fore_denseFlow,"inter_fore_denseflow");
//
    ////////////warping images to the reference frame///////////////////
            Mat warpedFrame=imgWarpFlow(im2, back_denseFlow);
            warpedToReference.push_back(warpedFrame.clone());
            //imshow("warped image",warpedFrame);
        }
        else{
            Mat refer_grey=video_input[reference_number].clone();
            warpedToReference.push_back(refer_grey.clone());
            back_flowfields.push_back(Mat::zeros(refer_grey.rows,refer_grey.cols,CV_32FC2));
            fore_flowfields.push_back(Mat::zeros(refer_grey.rows,refer_grey.cols,CV_32FC2));
        }
        }
    }

void motion_initiliazor_iterative(const vector<Mat>& video_input, vector<Mat>& back_flowfields, vector<Mat>& fore_flowfields, vector<Mat>& warpedToReference){
    int back_ransacThre=1;
    int fore_ransacThre=1;
    vector<Mat> backfields_iterative;
    vector<Mat> forefields_iterative;

    Mat im1, im2;
    Mat back_denseFlow, fore_denseFlow, back_iterFLow, fore_iterFlow;
    //flow: 0<-1<-2
    for (size_t frame_i=0; frame_i<reference_number; frame_i++){
        im1=video_input[frame_i+1].clone();
        im2=video_input[frame_i].clone();

        initila_motion_decompose(im1, im2, back_denseFlow, fore_denseFlow, back_ransacThre, fore_ransacThre);
        backfields_iterative.push_back(back_denseFlow.clone());
        forefields_iterative.push_back(fore_denseFlow.clone());
        }

    backfields_iterative.push_back(Mat::zeros(im2.rows,im2.cols,CV_32FC2));
    forefields_iterative.push_back(Mat::zeros(im2.rows,im2.cols,CV_32FC2));
    //flow: 2->3->4
    for (size_t frame_i=reference_number; frame_i<(frameNumber-1); frame_i++){
        im1=video_input[frame_i].clone();
        im2=video_input[frame_i+1].clone();

        initila_motion_decompose(im1, im2, back_denseFlow, fore_denseFlow, back_ransacThre, fore_ransacThre);
        backfields_iterative.push_back(back_denseFlow.clone());
        forefields_iterative.push_back(fore_denseFlow.clone());
//        colorFlow(back_denseFlow,"inter_back_denseflow");
//        colorFlow(fore_denseFlow,"inter_fore_denseflow");
        }
//
////////////warping images to the reference frame///////////////////
    for (size_t frame_i=0; frame_i<frameNumber; frame_i++){
        im2=video_input[frame_i].clone();
        back_denseFlow=Mat::zeros(im2.rows,im2.cols,CV_32FC2);//accumulate flow to the reference frame by iterative warping
        fore_denseFlow=Mat::zeros(im2.rows,im2.cols,CV_32FC2);

        if(frame_i==reference_number){
            warpedToReference.push_back(im2.clone());
            back_flowfields.push_back(back_denseFlow.clone());
            fore_flowfields.push_back(fore_denseFlow.clone());
        }
        else
            {for(int ii=0; ii<abs(int(reference_number)-int(frame_i)); ii++){
                int itera_ii = (int(reference_number)-int(frame_i))>0? (frame_i+ii) : (frame_i-ii);
                back_iterFLow=backfields_iterative[itera_ii].clone();
                fore_iterFlow=forefields_iterative[itera_ii].clone();
                ////sometimes foreground flow is better for reflections
                //Mat warped=imgWarpFlow(im2, fore_iterFlow);
                Mat warped=imgWarpFlow(im2, back_iterFLow);
                im2=warped.clone();
                cout<<frame_i<<reference_number<<itera_ii<<endl;

                if (ii>0){
                    back_denseFlow=addFlow(back_denseFlow,back_iterFLow);
                    fore_denseFlow=addFlow(fore_denseFlow,fore_iterFlow);
                }
                else{
                    back_denseFlow=back_iterFLow.clone();
                    fore_iterFlow=fore_iterFlow.clone();
                }
            }
            warpedToReference.push_back(im2.clone());
            back_flowfields.push_back(back_denseFlow.clone());
            fore_flowfields.push_back(fore_denseFlow.clone());
            colorFlow(back_denseFlow,"inter_back_denseflow");
            colorFlow(fore_denseFlow,"inter_fore_denseflow");
            }
        }

        }

void mot_decom_irls(const vector<Mat>& input_sequence, Mat& backgd_comp, Mat& obstruc_comp, Mat& alpha_map, const vector<Mat>& back_flowfields, const vector<Mat>& fore_flowfields, int nOuterFPIterations){
    int width = backgd_comp.cols;
    int height = backgd_comp.rows;
    int npixels = width*height;
    int nInnerFPIterations=10;

    double lambda_1 = 1;
    double lamdba_2 = 0.1;
    double lambda_3 = 3000;
    double lambda_4 = 0.5;

    double varepsilon = pow(0.001,2);
    Mat backgd_dx, backgd_dy, obstruc_dx, obstruc_dy;
    int deriv_ddepth = CV_64F;

    double omega_1[input_sequence.size()][npixels]={0};
    double omega_2[npixels]={0};
    double omega_3[npixels]={0};



    for(int ocount=0; ocount<nOuterFPIterations; ocount++){
        //compute gradients of current background and occlusion component
        Sobel(backgd_comp, backgd_dx, deriv_ddepth, 1, 0);
        Sobel(backgd_comp, backgd_dy, deriv_ddepth, 0, 1);
        Sobel(obstruc_comp, obstruc_dx, deriv_ddepth, 1, 0);
        Sobel(obstruc_comp, obstruc_dy, deriv_ddepth, 0, 1);
        //compute derivative denominators (weights)
        for(int icount =0; icount< nInnerFPIterations; icount++){
            //calculate the weights
            for(size_t tt=0; tt<input_sequence.size(); tt++){
                Mat img=input_sequence[tt];
                Mat back_flow=back_flowfields[tt];
                Mat obstruc_flow=fore_flowfields[tt];
                Mat temp = img-imgWarpFlow(obstruc_comp,obstruc_flow)-imgWarpFlow(alpha_map,obstruc_flow).mul(imgWarpFlow(backgd_comp,back_flow));
                for(int i=0; i<npixels;i++){
                    omega_1[tt][i]=1/sqrt(temp.at<double>(i)*temp.at<double>(i)+varepsilon);
                }
            }
            for(int i=0;i<npixels;i++){
                omega_2[i] = 1/sqrt(backgd_dx.at<double>(i)*backgd_dx.at<double>(i)+backgd_dy.at<double>(i)*backgd_dy.at<double>(i)+varepsilon);
                omega_3[i] = 1/sqrt(obstruc_dx.at<double>(i)*obstruc_dx.at<double>(i)+obstruc_dy.at<double>(i)*obstruc_dy.at<double>(i)+varepsilon);
            }

        }



    }

}

#include"edgeflow.h"
using namespace cv;

void computeCost(Mat im1,       // first image
		Mat im2,       // second image
        Mat im1_edge, // edge of the first image
        Mat im2_edge, // edge of the second image
		MRF::CostVal *&costData,   // computed cost volume
		int patch, //patch size
		int nL)               // number of labels
{
    int width = im1.cols, height = im1.rows;
    Mat edge_Locations1, label_Locations;

    //stretch into 1D array->one raw
    //Mat im1_array=im1.reshape(0,1);
    //Mat im2_array=im2.reshape(0,1);
    //Mat im1edge_array=im1_edge.reshape(0,1);


    //findNonZero(im1edge_array, edge_Locations1);
    //int edgeNumber = edge_Locations1.total();

    findNonZero(im1_edge, edge_Locations1); //edge in im1
    int edgeNumber = edge_Locations1.total();

    findNonZero(im2_edge, label_Locations); //edge in im2

    //Mat im1_edgex= Mat::zeros(1,edgeNumber,CV_8UC1);  //I1x
    //Mat im2_edgexd= Mat::zeros(1,edgeNumber,CV_8UC1); //I2(x+d)

    costData = new MRF::CostVal[width * height * nL];
    int dataIndex = 0;
    for (int i = 0; i < edgeNumber; i++ ) {
         //im1_edgex.at<unsigned char>(0,i)=im1_array.at<unsigned char>(edge_Locations1.at<Point>(i));

         Point2i x1 = edge_Locations1.at<Point>(i);

         //cout<<edge_Locations1.at<Point>(i)<<endl;
         for (int d = 0; d < nL; d++){
                Point2i x2 = label_Locations.at<Point>(d);
                double diffCost = 1;

                //cout<<label_Locations.at<Point>(i)<<endl;

                /*if (edge_Locations1.at<Point>(i).x+d<(width*height)){
                    //im2_edgexd.at<unsigned char>(0,i)=im2_array.at<unsigned char>(edge_Locations1.at<Point>(i).x+d,edge_Locations1.at<Point>(i).y);
                }
                else{
                    im2_edgexd.at<unsigned char>(0,i)=im2_array.at<unsigned char>(edge_Locations1.at<Point>(i).x+d-(width*height),edge_Locations1.at<Point>(i).y);
                    cout<<edge_Locations1.at<Po
                }*/
                // The cost of edge pixel i and label d is stored at dsi[i*nl+d]

                ncc_patch (im1, im2, x1, x2, patch, diffCost);
                costData[dataIndex++] = diffCost;


         }
         //cout << int(im1_edgex.at<unsigned char>(0,i)) << endl ;
    }

//calculate the NCC difference between patch around pix1 on im1 and pix2 on im2





}

void ncc_patch (Mat im1,
                Mat im2,
                Point2i pix1,
                Point2i pix2,
                int patch,
                double &ncc){

         Point patch_leftup1, patch_rightdow1, patch_leftup2, patch_rightdow2;
         //int patch_size = 2*patch+1;

         patch_leftup1 = Point(pix1.x - patch, pix1.y - patch);
         patch_rightdow1 = Point(pix1.x + patch, pix1.y + patch);
         patch_leftup2 = Point(pix2.x - patch, pix2.y - patch);
         patch_rightdow2 = Point(pix2.x + patch, pix2.y + patch);

         Mat patch1=im1(Rect(patch_leftup1, patch_rightdow1));
         Mat patch2=im1(Rect(patch_leftup2, patch_rightdow2));

         //Scalar mean1, mean2, stddev1, stddev2;
         /*
         meanStdDev(patch1, mean1, stddev1);
         meanStdDev(patch2, mean2, stddev2);

         double m1, m2, sd1, sd2;

         m1=mean1[0];
         m2=mean2[0];
         sd1=stddev1[0];
         sd2=stddev2[0];

        double covar = (patch1 - m1).dot(patch2 - m2) / (2*patch+1);
        double correl = covar / (sd1 * sd2);
         */
        Mat m1, m2;
        patch1.convertTo(m1, CV_32F);
        patch2.convertTo(m2, CV_32F);

        Mat correl;
        matchTemplate(m1, m2, correl, CV_TM_CCOEFF_NORMED);

        ncc= 1-correl.at<float>(0,0);
}

void computeCues(Mat im1, Mat im1_edge, MRF::CostVal *&hCue, MRF::CostVal *&vCue
		 ) {

        int width = im1.cols, height = im1.rows;
        hCue = new MRF::CostVal[width * height];
        vCue = new MRF::CostVal[width * height];

        int n=0;
        Mat edge_Locations1;
        findNonZero(im1_edge, edge_Locations1); //edge in im1
        int edgeNumber = edge_Locations1.total();

		 }

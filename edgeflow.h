//compute data term cost
#include <opencv2/opencv.hpp>
#include"opencv2/core/core.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mrf.h"

void computeCost(cv::Mat im1_grey, cv::Mat im2_grey, cv::Mat im1_edge, cv::Mat im2_edge, MRF::CostVal *&cData, int patch, int nL);

//ncc patch comparison
void ncc_patch (cv::Mat im1, cv::Mat im2, cv::Point2i pix1, cv::Point2i pix2, int patch, double &ncc);

void computeCues(cv::Mat im1, cv::Mat im1_edge, MRF::CostVal *&hCue, MRF::CostVal *&vCue);

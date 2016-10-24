/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */

#pragma once

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

#include <cudaPcl/depthGuidedFilter.hpp>
#include <cudaPcl/normalExtractSimpleGpu.hpp>
#include <cudaPcl/cv_helpers.hpp>

namespace cudaPcl {

/*
 * OpenniSmoothNormalsGpu smoothes the depth frame using a guided filter and
 * computes surface normals from it also on the GPU.
 *
 * Needs the focal length of the depth camera f_d and the parameters for the
 * guided filter eps as well as the filter size B.
 */
class OpenniSmoothNormalsGpu
{
  public:
  OpenniSmoothNormalsGpu(double f_d, double eps, uint32_t B, bool compress=false)
    : eps_(eps), B_(B), f_d_(f_d),
    depthFilter(NULL), normalExtract(NULL), compress_(compress)
  { };

  ~OpenniSmoothNormalsGpu() {
    if(normalExtract) delete normalExtract;
  };

  void depth_cb(const uint16_t * depth, uint32_t w, uint32_t h)
  {
    if(w==0 || h==0) return;

    if(!depthFilter)
    {
      depthFilter = new DepthGuidedFilterGpu<float>(w,h,eps_,B_);
      normalExtract = new NormalExtractSimpleGpu<float>(f_d_,w,h,compress_);
    }
    cv::Mat dMap = cv::Mat(h,w,CV_16U,const_cast<uint16_t*>(depth));

//    Timer t;
    depthFilter->filter(dMap);

//    t.toctic("smoothing");
    normalExtract->computeGpu(depthFilter->getDepthDevicePtr());
//    t.toctic("normals");
    normals_cb(normalExtract->d_normalsImg(), normalExtract->d_haveData(),w,h);
//    t.toctic("normals callback");
    if(compress_)
    {
      int32_t nComp =0;
      normalsComp_ = normalExtract->normalsComp(nComp);
      std::cout << "# compressed normals " << nComp << std::endl;
    }
  };

  /* callback with smoothed normals
   *
   * Note that the pointers are to GPU memory as indicated by the "d_" prefix.
   */
  void normals_cb(float* d_normalsImg, uint8_t* d_haveData,
      uint32_t w, uint32_t h)
  {
    if(w==0 || h==0) return;
    boost::mutex::scoped_lock updateLock(updateModelMutex);
    normalsImg_ = normalExtract->normalsImg();
    this->update_ = true;
  };

  static cv::Mat colorizeDepth(const cv::Mat& dMap, float min, float max);
  static cv::Mat colorizeDepth(const cv::Mat& dMap);

  void visualizeD();
  void visualizePC();
  cv::Mat getNormals();

  protected:
  double eps_;
  uint32_t B_;
  double f_d_;
  DepthGuidedFilterGpu<float> * depthFilter;
  NormalExtractSimpleGpu<float> * normalExtract;
  bool compress_;
  cv::Mat normalsImg_;
  cv::Mat nIRGB_;
  cv::Mat normalsComp_;

  bool update_;
  boost::mutex updateModelMutex;
  cv::Mat dColor_;
};

// ------------------------ impl -----------------------------------------
cv::Mat OpenniSmoothNormalsGpu::colorizeDepth(const cv::Mat& dMap, float min,
    float max)
{
//  double Min,Max;
//  cv::minMaxLoc(dMap,&Min,&Max);
//  cout<<"min/max "<<min<<" " <<max<<" actual min/max "<<Min<<" " <<Max<<endl;
  cv::Mat d8Bit = cv::Mat::zeros(dMap.rows,dMap.cols,CV_8UC1);
  cv::Mat dColor;
  dMap.convertTo(d8Bit,CV_8UC1, 255./(max-min));
  cv::applyColorMap(d8Bit,dColor,cv::COLORMAP_JET);
  return dColor;
}

cv::Mat OpenniSmoothNormalsGpu::colorizeDepth(const cv::Mat& dMap)
{
  double min,max;
  cv::minMaxLoc(dMap,&min,&max);
//  cout<<" computed actual min/max "<<min<<" " <<max<<endl;
  cv::Mat d8Bit = cv::Mat::zeros(dMap.rows,dMap.cols,CV_8UC1);
  cv::Mat dColor;
  dMap.convertTo(d8Bit,CV_8UC1, 255./(max-min));
  cv::applyColorMap(d8Bit,dColor,cv::COLORMAP_JET);
  return dColor;
}

cv::Mat OpenniSmoothNormalsGpu::getNormals()
{
	return normalsImg_.clone();
}

void OpenniSmoothNormalsGpu::visualizeD()
{
  if (this->depthFilter)
  {
    cv::Mat dSmooth = this->depthFilter->getOutput();
    this->dColor_ = colorizeDepth(dSmooth,0.3,4.0);
    cv::imshow("d",dColor_);
//    cv::Mat dNans = dSmooth.clone();
//    showNans(dNans);
//    cv::imshow("depth Nans",dNans);
  }
};

void OpenniSmoothNormalsGpu::visualizePC()
{
  if (normalsImg_.empty() || normalsImg_.rows == 0 || normalsImg_.cols
      == 0) return;
  cv::Mat nI (normalsImg_.rows,normalsImg_.cols, CV_8UC3);
//  cv::Mat nIRGB(normalsImg_.rows,normalsImg_.cols,CV_8UC3);
  normalsImg_.convertTo(nI, CV_8UC3, 127.5,127.5);
  cv::cvtColor(nI,nIRGB_,CV_RGB2BGR);
  cv::imshow("normals",nIRGB_);
  if (compress_)  cv::imshow("dcomp",normalsComp_);

  if (false) {
    // show additional diagnostics
    std::vector<cv::Mat> nChans(3);
    cv::split(normalsImg_,nChans);
    cv::Mat nNans = nChans[0].clone();
    showNans(nNans);
    cv::imshow("normal Nans",nNans);
    cv::Mat haveData = normalExtract->haveData();
    cv::imshow("haveData",haveData*200);
  }
}

} // namespace cudaPcl

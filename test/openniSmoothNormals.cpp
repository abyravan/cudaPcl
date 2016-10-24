/* Copyright (c) 2014, Julian Straub <jstraub@csail.mit.edu>
 * Licensed under the MIT license. See the license file LICENSE.
 */
// Modified by Arun to add PCL visualization

#include <iostream>
#include <string>

// Utilities and system includes
//#include <helper_functions.h>
#include <boost/program_options.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <Eigen/Dense>
#include <cudaPcl/openniSmoothNormalsGpu.hpp>

// PCL
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>

namespace po = boost::program_options;
using namespace Eigen;
using std::cout;
using std::endl;

int main (int argc, char** argv)
{
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("f_d,f", po::value<double>(), "focal length of depth camera")
        ("eps,e", po::value<double>(), "sqrt of the epsilon parameter of the guided filter")
        ("B,b", po::value<int>(), "guided filter windows size (size will be (2B+1)x(2B+1))")
        ("compress,c", "compress the computed normals")
        //    ("out,o", po::value<std::string>(), "output path where surfae normal images are saved to")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    double f_d = 525.0;
    double eps = 0.2*0.2;
    int32_t B = 10;
    bool compress = false;
    if(vm.count("f_d")) f_d = vm["f_d"].as<double>();
    if(vm.count("eps")) eps = vm["eps"].as<double>();
    if(vm.count("B")) B = vm["B"].as<int>();
    if(vm.count("compress")) compress = true;

    findCudaDevice(argc,(const char**)argv);
    cudaPcl::OpenniSmoothNormalsGpu v(f_d, eps, B, compress);

    if(true)
    {
        // load a specific image and process
        cv::Mat depth = cv::imread("test/table_0_d.png",CV_LOAD_IMAGE_ANYDEPTH);
        for(uint32_t t=0; t<1; ++t)
        {
            v.depth_cb((uint16_t*)depth.data,depth.cols,depth.rows);
            v.visualizeD();
            v.visualizePC();
            cv::waitKey(30);
        }

        // Get normals image
        cv::Mat normalsImg = v.getNormals();

        // Setup cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>(depth.cols, depth.rows));
        pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>(depth.cols, depth.rows));
        float depthres = 0.001;
        for (int r = 0; r < depth.rows; r++)
        {
            for (int c = 0; c < depth.cols; c++)
            {
                // Project to 3D
                float z = depth.at<ushort>(r,c) * depthres;
                float x = (c - 0.5*depth.cols)/f_d;
                float y = (r - 0.5*depth.rows)/f_d;
                x *= z;
                y *= z;

                // PCL cloud
                cloud->at(c,r) = pcl::PointXYZ(x,y,z);
                normals->at(c,r) = pcl::Normal(normalsImg.at<cv::Vec3f>(r,c)[0],
                                               normalsImg.at<cv::Vec3f>(r,c)[1],
                                               normalsImg.at<cv::Vec3f>(r,c)[2]);
            }
        }

        // visualize normals
        pcl::visualization::PCLVisualizer viewer("PCL Viewer");
        viewer.setBackgroundColor (0.3, 0.3, 0.3);
        viewer.addPointCloud(cloud,"cloud");
        viewer.addPointCloudNormals<pcl::PointXYZ,pcl::Normal>(cloud, normals,10,0.05,"normals");
        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "normals");

        // Display viewer
        while (!viewer.wasStopped ())
        {
            viewer.spinOnce ();
        }
    }

    // run the grabber
    // v.run ();
  
    // cout << cudaDeviceReset() << endl;
    return (0);
}

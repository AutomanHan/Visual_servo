#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;
int main()
{
	VideoCapture cap(0);
	Mat frame;

	Ptr<SVM> svm=SVM::load("./parts_svm.xml");
	if(svm->empty())
	{
		cout<<"load svm detector failed!!!"<<endl;
		return 0;
	}

	//特征响亮位数
	int DescriptorDim; 
	//特征向量的维数，即HOG描述子的维数  
	DescriptorDim = svm->getVarCount();
	//获取svecsmat，元素类型为float
	cv::Mat svecsmat = svm->getSupportVectors();
	//特征向量维数
	int svdim = svm->getVarCount();
	int numofsv = svecsmat.rows;
	//alphamat和svindex必须初始化，否则getDecisionFunction()函数会报错
	cv::Mat alphamat = cv::Mat::zeros(numofsv, svdim, CV_32F);
	cv::Mat svindex = cv::Mat::zeros(1, numofsv, CV_64F);
 
	cv::Mat Result;
	double rho = svm->getDecisionFunction(0, alphamat, svindex);
	//将alphamat元素的数据类型重新转成CV_32F
	alphamat.convertTo(alphamat, CV_32F);
	Result = -1 * alphamat * svecsmat;
 
	std::vector<float> vec;
	for (int i = 0; i < svdim; ++i)
	{
		vec.push_back(Result.at<float>(0, i));
	}
	vec.push_back(rho);
	//Mat img=imread("/home/hanc/Code/visual_servo/hog/img_label/class_2/class_1.jpg",0);
	//resize(img, img, Size(96,96));
	HOGDescriptor hog;
	hog.winSize=Size(64,128);
	vector<float> descriptors;
	/*
	String class1_path="/home/hanc/Code/visual_servo/hog/img_label/class_1";
	String class2_path="/home/hanc/Code/visual_servo/hog/img_label/class_2";
	bool addPath=false;
	vector<String> name_class1;
	vector<String> name_class2;
	glob(class1_path, name_class1, addPath);
	glob(class2_path, name_class2, addPath);
	
	int numclass_1=name_class1.size();
	int numclass_2=name_class2.size();
	int response=0;
	for(int i=0;i<numclass_1;i++)
	{
		//cout<<"image path:"<<name_class1[i]<<endl;
		Mat img=imread(name_class1[i],0);
		resize(img, img, Size(96,96));
		hog.compute(img, descriptors, Size(8,8),Size(0,0));
		response=(int)svm->predict(descriptors);
		cout<<i<<".jpg分类结果"<< response<< endl;
	}
	for(int i=0;i<numclass_2;i++)
	{
		//cout<<"image path:"<<name_class1[i]<<endl;
		Mat img=imread(name_class2[i],0);
		resize(img, img, Size(96,96));
		hog.compute(img, descriptors, Size(8,8),Size(0,0));
		response=(int)svm->predict(descriptors);
		cout<<i<<".jpg分类结果"<< response<< endl;
	}

	*/
	//hog.compute(img, descriptors, Size(8,8),Size(0,0));
	//int response = (int)svm->predict(descriptors);
	
	cout<<"setting"<<endl;
	hog.setSVMDetector(vec);
	cout<<"set successfully"<<endl;

	RNG g_rng(12345); //用于生成随机数

	
	
	while(1)
	{
		cap>>frame;
		imshow("src",frame);
		char key=waitKey(10);
		if(key=='z'||key=='Z')
		{
			Mat img_gray;
			vector<Rect> found;
			cvtColor(frame, img_gray, CV_BGR2GRAY);
			resize(img_gray, img_gray, Size(img_gray.rows, img_gray.rows*2));
			hog.detectMultiScale(img_gray, found, 0, Size(8,8), Size(0,0));
			cout<<"numbers of find square:"<<found.size()<<endl;
			for(int i=0; i<found.size();i++)
			{
				Rect r=found[i];
				rectangle(img_gray, r, Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255)));
			}
			imshow("parts:", img_gray);
			waitKey();

		}
		if(key=='q'||key=='Q')
			break;
	}

	return 0;
}
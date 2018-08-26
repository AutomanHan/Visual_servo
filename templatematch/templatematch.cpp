#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <iostream>
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
Mat roi;
Rect selection;
Point origin;
Point move_mouse;
bool selectObject;
bool trackObject=false;
bool draw_rect=false;
RNG g_rng(12345); //用于生成随机数
/*
image coordinate system in opencv 
     					  x
    o--------------------->
	 |	
	 |
	 |
	 |
	 |
	y|
*/


void onMouse(int event, int x, int y, int, void*)
{
	
	char temp[16];
	if(selectObject)
	{
		selection.x=origin.x;
		selection.y=origin.y;
		selection.width=abs(x-origin.x);
		selection.height=abs(y-origin.y);

		selection &= Rect(0,0, roi.cols, roi.rows);

	}

	switch (event)
	{
		case CV_EVENT_LBUTTONDOWN:
			origin=Point(x,y);
			selection=Rect(x,y,0,0);
			selectObject=true;
			draw_rect=false;
			break;
		case CV_EVENT_LBUTTONUP:
			selectObject=false;
			if(selection.width>0 && selection.height>0)
				trackObject=true;
				draw_rect=true;
			break;
		default:
			break;
	}
}
bool g_run=false;
void onTrackbarSlide(int pos, void *)
{
	g_run=true;
}

int main()
{
	VideoCapture cap(0);
	if(!cap.isOpened())
	{
		cout<<"open camera error"<<endl;
		return -1;
	}
	Mat frame;
	Mat img_gray;
	Mat parts;
	bool drawkeyP=false;
	char name_window[20]="roi_select";
	int hessien=100;
	namedWindow(name_window, WINDOW_AUTOSIZE);
	setMouseCallback(name_window, onMouse, 0);
	createTrackbar("hessien", name_window,&hessien, 1000, onTrackbarSlide);
	int length=0;

	while(1)
	{
		cap>>frame;
		cvtColor(frame, img_gray, CV_BGR2GRAY);
		//cout<<frame.size()<<endl;
		roi=img_gray(Rect(100,100,400, 300));
		//blur(roi, roi, Size(5,5));
		//threshold(roi, roi, 0,255, THRESH_OTSU+THRESH_BINARY);
		//imshow("src image", frame);
		//if(draw_rect)
		rectangle(roi, selection, cv::Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255)));
		parts=roi(selection);
		imshow(name_window, roi);
		//createTrackbar("length:\n", name_window, &length, 8, );
		if(draw_rect)
			imshow("parts", parts);
		char key=waitKey(10);
		if(key == 'Q' || key == 'q')
		{
			drawkeyP=true;
			cout<<"any keyboard"<<endl;
			//waitKey();
			Ptr<SURF> detector=SURF::create(hessien);
			vector<KeyPoint> keypoints;
			Mat surfDescriptor;
			detector->detect(parts, keypoints);
			detector->compute(parts, keypoints, surfDescriptor);
			cout<<"size of keypoints:"<<keypoints.size()<<endl;
			//cout<<"size of surfDescriptor:"<<surfDescriptor[1].size()<<endl;
			Mat img_keypoints;
			drawKeypoints(parts, keypoints, img_keypoints);
			imshow("keypoint",img_keypoints);
			waitKey();
			//break;
		}
	}
	/*
	//提取特征点
	SurfFeatureDetector surfDetector();
	vector<KeyPoint> keyPoint1, keyPoint2;
	surfDetector.detect(parts, keyPoint1);

	//特征描述子
	SurfDescriptorExtractor surfDescriptor;
	Mat imageDesc1;
	surfDescriptor.compute(parts, keypoint1, imageDesc1);
	*/


	//while(drawkeyP)
	//{
		
	//	char key=waitKey(10);
	//	if(key=='q'||key=='Q')
	//	{
	//	}	
	//}
	

	return 0;
}
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;
bool selectObject=false;
bool draw_rect=false;
bool sample=false;
Point origin;
Rect selection;
Mat roi;
RNG g_rng(12345); //用于生成随机数
Mat frame;
void onMouse(int event, int x, int y, int, void*)
{
	//bool selecti=false;
	char temp[16];
	if(selectObject)
	{
		selection.x=origin.x;
		selection.y=origin.y;
		selection.width=abs(x-origin.x);
		selection.height=abs(y-origin.y);

		selection &= Rect(0,0, frame.cols, frame.rows);

	}

	switch (event)
	{
		case CV_EVENT_LBUTTONDOWN:
			origin=Point(x,y);
			selection=Rect(x,y,0,0);
			selectObject=true;
			draw_rect=false;
			cout<<"mouse left down"<<endl;
			break;
		case CV_EVENT_LBUTTONUP:
			
			if(selection.width>0 && selection.height>0)
				draw_rect=true;
				cout<<"mouse left up"<<endl;
				sample=true;
				selectObject=false;
			break;
		default:
			break;
	}
}
int main()
{	
	VideoCapture cap(0);
	if(!cap.isOpened())
	{
		cout << "can't open camera"<<endl;
		return 0;
	}
	char name_window[20]="roi_select";
	char source_name[20]="source";
	namedWindow(source_name, WINDOW_AUTOSIZE);
	setMouseCallback(source_name, onMouse, 0);
	//Mat frame;
	Mat img_gray;
	Mat img_sobel;
	Mat img_threshold;
	vector<vector<Point> > contours;
	
	int num=0, count=0;
	
	Ptr<SVM> svm = SVM::load("./parts_svm.xml");
	HOGDescriptor hog;
	hog.winSize=Size(96,96);
	vector<float> descriptors;
	int response=5;
			Rect brect;
			Rect part_rect;
	while(1)
	{
		cap>>frame;
		cvtColor(frame, img_gray, CV_BGR2GRAY);
		blur(img_gray, img_gray, Size(3,3));
		//frame.copyTo();
		rectangle(img_gray, selection, Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255)));
		imshow(source_name,img_gray);
		roi=img_gray(selection);
		if(draw_rect==true)
		{
			imshow(name_window, roi);
			Sobel(roi, img_sobel, CV_8U, 1, 1, 3, 1, 0, BORDER_DEFAULT);
			//imshow("sobel", img_sobel);
			threshold(img_sobel, img_threshold, 0, 255, THRESH_OTSU + THRESH_BINARY );
			
			//形态学操作
			Mat element = getStructuringElement(MORPH_RECT , Size(21, 21));
			morphologyEx(img_threshold, img_threshold, MORPH_CLOSE, element);
			imshow("binary", img_threshold);
			findContours(img_threshold, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
			std::vector<std::vector<Point> >::iterator itc=contours.begin();
			vector<RotatedRect> rects;
			
			//cout<<"test1"<<endl;
			if(sample==true)
			{
			while(itc!=contours.end())
			{
				RotatedRect mr=minAreaRect(*itc);
				brect=mr.boundingRect();
				
				if(brect.width*brect.height>2000)
				{
					num++;
					cout<<"test2"<<endl;
					
					cout<<"test3"<<endl;
					Mat rect_part;
					cout<<"brect.x:"<<brect.x<<endl;
					cout<<"brect.y:"<<brect.y<<endl;
					cout<<"brect.width:"<<brect.width<<endl;
					cout<<"brect.cols:"<<brect.size()<<endl;
					cout<<"brect.height:"<<brect.height<<endl;
					cout<<"roi.cols:"<<roi.cols<<endl;
					cout<<"roi.rows:"<<roi.rows<<endl;
					cout<<"img_threshold.cols:"<<img_threshold.cols<<endl;
					cout<<"img_threshold.rows:"<<img_threshold.rows<<endl;
					//cout<<"brect.rows:"<<brect.rows<<endl;
					if(brect.x>0 && brect.y>0 && brect.x+brect.width<roi.cols && brect.y+brect.height<roi.rows)
					{
						rect_part= roi(brect);
						resize(rect_part, rect_part, Size(96,96));
						hog.compute(rect_part, descriptors, Size(8,8), Size(0,0));
						response=(int)svm->predict(descriptors);
						imshow("countor:", rect_part);
						//stringstream ss(stringstream::in | stringstream::out);
						//ss<<"class_"<<response;
						part_rect=brect;
						cout<<"sample successful"<<endl;
						//imwrite(ss.str(),rect_part);
					}
				}
				++itc;
			}
			
			sample=false;
			}
			char ss[20];
						cout<<"class_"<<response<<endl;
						cout<<"brect.x:"<<part_rect.x<<"parts.y"<<part_rect.y<<"part_rect.width"<<part_rect.width<<"part_rect.height"<<part_rect.height<<endl;
						sprintf(ss,"class_%d",response);
						putText(roi, ss, Point(part_rect.x, part_rect.y), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255)));
						rectangle(roi, part_rect, Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255)));
						imshow("parts", roi);


		}

		char key = waitKey(10);
		if(key=='z'||key=='Z')
		{
			imwrite("src.jpg", img_gray);
			imwrite("roi.jpg", roi);
		}

		if(key=='q'|| key=='Q')
		{
			break;
		}
	}
	return 0;
}
#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "BorderMatting.h"
#include <vector>
#include <unordered_map>
#include <math.h>
using namespace std;
using namespace cv;

struct point{
	int x;
	int y;
};

struct para_point{
	point p;
	int index;
	int section;
	double delta;
	double sigma;
};

struct inf_point{
	point p;
	int dis;
	int area;
};

struct dands{
	int delta;
	int sigma;
};

typedef vector<double[30][10]> Energyfunction;
typedef vector<dands[30][10]> Record;
typedef vector<para_point> Contour;
typedef unordered_map<int, inf_point> Strip;

/*轮廓上视为相邻的8个点*/
#define nstep 8
const int nx[nstep] = { 0, 1, 0, -1, -1, -1, 1, 1 };
const int ny[nstep] = { 1, 0, -1, 0, -1, 1, -1, 1 };

#define COE 10000

#define stripwidth 6

#define L 20

/*欧式距离为1的相邻点*/
#define rstep 4
const int rx[rstep] = {0,1,0,-1};
const int ry[rstep] = {1,0,-1,0};

#define MAXNUM 9999999;

#define sigmaLevels  15
#define deltaLevels  11

class BorderMatting
{
public:
	BorderMatting();
	~BorderMatting();
	void borderMatting(const Mat& oriImg, const Mat& mask, Mat& borderMask);
private:
	void ParameterizationContour(const Mat& edge);
	void dfs(int x, int y, const Mat& mask, Mat& amask);
	void StripInit(const Mat& mask);
	void EnergyMinimization(const Mat& oriImg, const Mat& mask);
	inline double Vfunc(double ddelta, double dsigma)
	{
		return (lamda1*pow(ddelta, 2.0) + lamda2*pow(dsigma, 2.0))/200;
	}
	void init(const Mat& img);
	double Dfunc(int index, point p, double uf, double ub, double cf, double cb, double delta, double sigma, const Mat& gray);
	void CalculateMask(Mat& bordermask, const Mat& mask);
	void display(const Mat& oriImg,const Mat& mask);

	const int lamda1 = 50;
	const int lamda2 = 1000;
	int sections; //独立轮廓个数
	int rows, cols; 
	int areacnt; //区域个数（即轮廓上点的个数）
	int tot;
	Contour contour; //轮廓
	Strip strip; //条带
	double ef[5000][deltaLevels][sigmaLevels];
	dands rec[5000][deltaLevels][sigmaLevels];
	vector<dands> vecds;
};

#include "BorderMatting.h"


BorderMatting::BorderMatting()
{
}

BorderMatting::~BorderMatting()
{
}

inline bool outrange(int x, int l, int r)
{
	if (x<l || x>r)
		return true;
	else
		return false;
}

/*变量初始化*/
void BorderMatting::init(const Mat& img)
{
	rows = img.rows;
	cols = img.cols;
	sections = 0;
	areacnt = 0;
	tot = 0;
	contour.clear();
	strip.clear();
	vecds.clear();
}

/*边缘检测*/
void BorderDetection(const Mat& img, Mat& rs)
{
	Mat edge;
	Canny(img, edge, 3, 9, 3);
	edge.convertTo(rs, CV_8UC1);
}

/*深搜遍历轮廓*/
void BorderMatting::dfs(int x, int y, const Mat& edge, Mat& color)
{
	color.at<uchar>(x, y) = 255;//标记已遍历
	para_point pt;
	pt.p.x = x; pt.p.y = y; //坐标
	pt.index = areacnt++;//给轮廓上每一个点分配独立index
	pt.section = sections;//所属轮廓
	contour.push_back(pt); //放入轮廓vector
	for (int i = 0; i < nstep; i++)//枚举(x,y)相邻点
	{
		int zx = nx[i];
		int zy = ny[i];
		int newx = x + zx;
		int newy = y + zy;
		if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1)) //超出图像范围
			continue;
		if (edge.at<uchar>(newx,newy) == 0 || color.at<uchar>(newx,newy) != 0) //不是轮廓上的点，或者已经被遍历过
			continue;
		dfs(newx,newy,edge,color);//从(newx,newy)出发，继续深搜遍历轮廓
	}
}

/*轮廓参数化*/
void BorderMatting::ParameterizationContour(const Mat& edge)
{
	int rows = edge.rows;
	int cols = edge.cols;
	sections = 0; //独立轮廓的个数
	areacnt = 0; //以轮廓上不同点为中心的区域个数（即轮廓上点的个数）
	Mat color(edge.size(), CV_8UC1, Scalar(0));//遍历标记
	bool flag = false;
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			if (edge.at<uchar>(i, j) != 0)//(i,j)是轮廓上的点
			{
				if (color.at<uchar>(i, j) != 0)//该轮廓已经被遍历过
					continue;
				/*(i,j)所在轮廓线还没有被遍历过*/
				dfs(i, j, edge, color);//深搜遍历轮廓
				sections++;//轮廓个数加1
			}	
}

/*条带区域化*/
void BorderMatting::StripInit(const Mat& mask)
{
	Mat color(mask.size(), CV_32SC1, Scalar(0));//遍历标记

	/*从轮廓出发，宽搜标记条带，标记条带所属区域————对应的中心轮廓点*/
	//初始化队列：加入轮廓上所有点
	vector<point> queue;
	for (int i = 0; i < contour.size(); i++)
	{
		inf_point ip;
		ip.p = contour[i].p; //坐标
		ip.dis = 0; //距离中心点的欧氏距离
		ip.area = contour[i].index; //所属区域
		strip[ip.p.x*COE + ip.p.y] = ip; //将点加入条带，key（hash）值为其坐标
		queue.push_back(ip.p); //将点加入队列
		color.at<int>(ip.p.x, ip.p.y) = ip.area+1; //遍历标记：区域号+1
	}
	//宽搜遍历条带
	int l = 0;
	while (l < queue.size())
	{
		point p = queue[l++]; //取出点
		inf_point ip = strip[p.x*COE+p.y]; //从strip中得到相关信息
		if (abs(ip.dis) >= stripwidth) //只遍历条带内的点
			break;
		int x = ip.p.x;
		int y = ip.p.y;
		for (int i = 0; i < rstep; i++)//枚举相邻点
		{
			int newx = x + rx[i];
			int newy = y + ry[i];
			if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1))//超出图像范围
				continue;
			inf_point nip;
			if (color.at<int>(newx, newy) != 0){//该点曾经被遍历过
				///*轮廓上每6个点取一个关键点。优先将条带分配给以这些关键点为中心的区域*/
				//if (ip.area % 6 != 0) //当前中心点不是关键点
				//	continue;
				//if ((color.at<int>(newx, newy) - 1) % 6 == 0) //当前点已属于关键点区域
				//	continue;
				//nip = strip[newx*COE+newy];
				continue;
			}
			else
			{
				nip.p.x = newx; nip.p.y = newy;
			}
			nip.dis = abs(ip.dis) + 1;//欧式距离+1
			if ((mask.at<uchar>(newx, newy) & 1) != 1 ) //属于背景
			{
				nip.dis = -nip.dis;
			}
			nip.area = ip.area;
			strip[nip.p.x*COE + nip.p.y] = nip; //加入条带
			queue.push_back(nip.p); //加入队列
			color.at<int>(newx, newy) = nip.area+1; //遍历标记：区域号+1
		}
	}
}

/*高斯密度函数*/
inline double Gaussian(double x, double delta, double sigma)
{
	const double PI = 3.14159;
	double e = exp(-(pow(x-delta,2.0)/(2.0*sigma)));
	double rs = 1.0 / (pow(sigma,0.5)*pow(2.0*PI, 0.5))*e;
	return rs;
}

inline double ufunc(double a,double uf,double ub)
{
	return (1.0 - a)*ub + a*uf;
}

inline double cfunc(double a, double cf,double cb)
{
	return pow(1.0 - a, 2.0)*cb + pow(a, 2.0)*cf;
}

/*sigmoid函数*/
inline double Sigmoid(double r, double delta, double sigma)
{
	double rs = -(r - delta) / sigma;
	rs = exp(rs);
	rs = 1.0 / (1.0 + rs);
	return rs;
}

inline double Dterm(inf_point ip, float I, double delta, double sigma, double uf, double ub, double cf, double cb )
{
	double alpha = Sigmoid((double)ip.dis / (double)stripwidth, delta, sigma);
	double D = Gaussian(I, ufunc(alpha, uf, ub), cfunc(alpha, cf, cb));
	D = -log(D) / log(2.0);
	return D;
}

/*计算term D*/
double BorderMatting:: Dfunc(int index, point p, double uf, double ub, double cf, double cb, double delta, double sigma, const Mat& gray)
{
	vector<inf_point> queue;
	map<int, bool> color;
	double sum = 0;
	inf_point ip = strip[p.x*COE + p.y]; //从strip中获取中心点信息
	sum += Dterm(ip, gray.at<float>(ip.p.x, ip.p.y),delta,sigma,uf,ub,cf,cb);
	queue.push_back(ip);//加入队列
	color[ip.p.x*COE + ip.p.y] = true;//标记遍历
	/*宽搜遍历以p为中心点的区域*/
	int l = 0;
	while (l < queue.size())
	{
		inf_point ip = queue[l++];
		if (abs(ip.dis) >= stripwidth)
			break;
		int x = ip.p.x;
		int y = ip.p.y;
		for (int i = 0; i < rstep; i++)//枚举相邻点
		{
			int newx = x + rx[i];
			int newy = y + ry[i];
			if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1)) //超出图像范围
				continue;
			if (color[newx*COE+newy])//已经遍历过
				continue;
			inf_point newip = strip[newx*COE+newy];//从strip中获取点的信息
			if (newip.area == index) //属于以p为中心点的区域
			{
				sum += Dterm(newip, gray.at<float>(newx, newy), delta, sigma, uf, ub, cf, cb);
			}
			queue.push_back(newip);//加入队列
			color[newx*COE + newy] = true;//标记遍历
		}
	}
	return sum;
}

/*计算L*L区域的前背景均值和方差*/
void calSampleMeanCovariance(point p, const Mat& gray, const Mat& mask, double& uf, double& ub, double& cf, double& cb)
{
	int len = L;
	double sumf=0, sumb=0;
	int cntf = 0, cntb = 0;
	int rows = gray.rows;
	int cols = gray.cols;
	//计算均值
	for (int x = p.x - len; x <= p.x + len; x++)
		for (int y = p.y - len; y <= p.y + len; y++)
			if  (!(outrange(x, 0, rows - 1) || outrange(y, 0, cols - 1)))
				{
					float g = gray.at<float>(x, y);
					if ((mask.at<uchar>(x, y) & 1) == 0) //背景
					{
						sumb += g;
						cntb++;
					}
					else //前景
					{
						sumf += g;
						cntf++;
					}
				}
	uf = (double)sumf / (double)cntf; //前景均值
	ub = (double)sumb / (double)cntb; //背景均值
	//计算方差
	cf = 0;
	cb = 0;
	for (int x = p.x - len; x <= p.x + len; x++)
		for (int y = p.y - len; y <= p.y + len; y++)
			if (!(outrange(x, 0, rows - 1) || outrange(y, 0, cols - 1)))
			{
				float g = gray.at<float>(x, y);
				if ((mask.at<uchar>(x, y) & 1) == 0) //背景
				{
					cb += pow(g - ub, 2.0);
				}
				else //前景
				{
					cf += pow(g - uf, 2.0);
				}
			}
	cf /= (double)cntf; //前景方差
	cb /= (double)cntb; //背景方差
}

/*通过level计算sigma*/
inline double sigma(int level)
{
	return 0.025*(level);
}

/*通过level计算delta*/
inline double delta(int level)
{
	return 0.1*level;
	//return -1.0 + 0.2*level;
}

/*能量最小化，动态规划求每个区域的delta和sigma*/
void BorderMatting::EnergyMinimization(const Mat& oriImg, const Mat& mask)
{
	//转换为灰度图
	Mat gray;
	cvtColor(oriImg, gray, COLOR_BGR2GRAY);
	gray.convertTo(gray,CV_32FC1,1.0/255.0);
	//能量最小化求每个区域的delta和sigma
	for (int i = 0; i < contour.size(); i++)//枚举轮廓上每一个点，即条带各区域中心点
	{
		para_point pp = contour[i];
		int index = pp.index;
		double uf,ub,cf,cb;
		//求L*L区域的前背景均值和方差
		calSampleMeanCovariance(pp.p,gray,mask,uf,ub,cf,cb);
		for (int d0 = 0; d0< deltaLevels; d0++) //枚举delta
			for (int s0 = 0; s0 < sigmaLevels; s0++) //枚举sigma
			{
				double sigma0 = sigma(s0);
				double delta0 = delta(d0);
				ef[index][d0][s0] = MAXNUM;
				//计算term D
				double D = Dfunc(index, pp.p, uf, ub, cf, cb, delta0, sigma0, gray);
				//计算能量方程:termD + termV
				if (index == 0)
				{
					ef[index][d0][s0] = D;
					continue;
				}
				//if (index % 6 != 0)//为了加快计算，非关键点不做枚举。d0,s0取与(index-1)时相同
				//{
				//	ef[index][d0][s0] = ef[index - 1][d0][s0] + D;
				//	dands ds;
				//	ds.delta = d0;
				//	ds.sigma = s0;
				//	rec[index][d0][s0] = ds;
				//	continue;
				//}
				for (int d1 = 0; d1 < deltaLevels; d1++)//枚举index-1时的delta
					for (int s1 = 0; s1 < sigmaLevels; s1++)//枚举index-1时的sigma
					{
						double delta1 = delta(d1);
						double sigma1 = sigma(s1);
						double Vterm = 0;
						if (contour[i - 1].section == pp.section)//与上一点属于同一轮廓
						{
							Vterm = Vfunc(delta0 - delta1, sigma0 - sigma1);
						}
						double rs = ef[index-1][d1][s1] + Vterm + D;
						if (rs < ef[index][d0][s0])
						{
							dands ds;
							ds.sigma = s1; ds.delta = d1;
							ef[index][d0][s0] = rs;
							rec[index][d0][s0] = ds;
						}
					}
			}
	}
	//找总能量最小值
	double minE = MAXNUM;
	dands ds;
	vecds = vector<dands>(areacnt);//记录每个区域的delta和sigma
	for (int d0 = 0; d0< deltaLevels; d0++)
		for (int s0 = 0; s0 < sigmaLevels; s0++)
		{
			if (ef[areacnt-1][d0][s0] < minE)
			{
				minE = ef[areacnt-1][d0][s0];
				ds.delta = d0;
				ds.sigma = s0;
			}
		}
	//记录总能量最小时，每个区域的delta和sigma
	vecds[areacnt-1]=ds;
	for (int i = areacnt - 2; i >= 0; i--)
	{
		dands ds0 = vecds[i + 1];
		dands ds = rec[i + 1][ds0.delta][ds0.sigma];
		vecds[i]=ds;
	}
}

/*调整alpha*/
inline double adjustA(double a)
{
	if (a < 0.01)
		return 0;
	if (a > 9.99)
		return 1;
	return a;
}

/*计算每个像素点的alpha*/
void BorderMatting::CalculateMask(Mat& bordermask, const Mat& mask)
{
	bordermask = Mat(mask.size(), CV_32FC1, Scalar(0));

	Mat color(mask.size(), CV_32SC1, Scalar(0));//遍历标记

	/*从轮廓出发，宽搜遍历图像，计算alpha*/
	//初始化队列：加入轮廓上所有点
	vector<inf_point> queue;
	for (int i = 0; i < contour.size(); i++)
	{
		inf_point ip;
		ip.p = contour[i].p; //坐标
		ip.dis = 0; //距离中心点的欧氏距离
		ip.area = contour[i].index; //所属区域
		queue.push_back(ip); //将点加入队列
		color.at<int>(ip.p.x, ip.p.y) = 1; //遍历标记
		//计算alpha
		dands ds = vecds[ip.area];
		double alpha = Sigmoid((double)ip.dis / (double)stripwidth, delta(ds.delta), sigma(ds.sigma));
		alpha = adjustA(alpha);//调整alpha
		bordermask.at<float>(ip.p.x, ip.p.y) = (float)alpha;
	}
	//宽搜遍历条带
	int l = 0;
	while (l < queue.size())
	{
		inf_point ip = queue[l++]; //取出点
		int x = ip.p.x;
		int y = ip.p.y;
		for (int i = 0; i < rstep; i++)//枚举相邻点
		{
			int newx = x + rx[i];
			int newy = y + ry[i];
			if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1))//超出图像范围
				continue;
			inf_point nip;
			if (color.at<int>(newx, newy) != 0)
				continue;
			nip.p.x = newx; nip.p.y = newy;
			nip.dis = abs(ip.dis) + 1;//欧式距离+1
			if ((mask.at<uchar>(newx, newy) & 1) != 1) //属于背景
			{
				nip.dis = -nip.dis;
			}
			nip.area = ip.area;
			queue.push_back(nip); //加入队列
			color.at<int>(newx, newy) = 1; //遍历标记
			//计算alpha
			dands ds = vecds[nip.area];
			double alpha = Sigmoid((double)nip.dis / (double)stripwidth, delta(ds.delta), sigma(ds.sigma));
			alpha = adjustA(alpha);//调整alpha
			bordermask.at<float>(nip.p.x, nip.p.y) = (float)alpha;
		}
	}
}

void BorderMatting::display(const Mat& oriImg, const Mat& borderMask)
{
	/*用mask遮罩处理前景*/
	vector<Mat> ch_img(3);
	vector<Mat> ch_bg(3);
	//分离前景三通道
	Mat img;
	oriImg.convertTo(img, CV_32FC3, 1.0 / 255.0);
	cv::split(img, ch_img);
	//分离背景三通道
	Mat bg = Mat(img.size(), CV_32FC3, Scalar(1.0, 1.0, 1.0));
	cv::split(bg, ch_bg);
	//mask遮罩处理
	ch_img[0] = ch_img[0].mul(borderMask) + ch_bg[0].mul(1.0 - borderMask);
	ch_img[1] = ch_img[1].mul(borderMask) + ch_bg[1].mul(1.0 - borderMask);
	ch_img[2] = ch_img[2].mul(borderMask) + ch_bg[2].mul(1.0 - borderMask);
	//合并三通道
	Mat res;
	cv::merge(ch_img, res);
	//显示结果
	Mat tem, tem2;
	resize(borderMask, tem2, Size(0, 0), 4, 4);
	resize(res, tem, Size(0, 0), 4, 4);
	imshow("result", tem);
	imshow("img", res);
	imshow("mask", tem2);
}

/*border matting*/
void BorderMatting::borderMatting(const Mat& oriImg, const Mat& mask, Mat& borderMask)
{
	/*初始化部分参数*/
	init(oriImg);
	
	/*mask轮廓检测*/
	Mat edge = mask & 1;
	edge.convertTo(edge, CV_8UC1, 255);
	BorderDetection(edge,edge);
	
	/*轮廓参数化*/
	ParameterizationContour(edge);
	
	/*条带区域化*/
	Mat tmask;
	mask.convertTo(tmask,CV_8UC1);
	StripInit(tmask);

	/*能量最小化，动态规划求每个区域的delta和sigma*/
	EnergyMinimization(oriImg, mask);

	/*计算遮罩 —— alpha mask*/
	CalculateMask(borderMask, mask);	//计算每个像素点的alpha
	GaussianBlur(borderMask, borderMask, Size(7, 7), 9);	//对alpha mask进行轻微高斯模糊

	/*显示border matting结果*/
	display(oriImg,borderMask);	
}



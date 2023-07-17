#include "Pixel.h"
#include <iostream>
#include <algorithm>
using namespace std;

void Pixel::add(int cnst){
	this->red = std::min(this->red + cnst, 255);
	this->green = std::min(this->green + cnst, 255);
	this->blue = std::min(this->blue + cnst, 255);
}

void Pixel::sub(int cnst){
	this->red = std::max(this->red - cnst, 0);
	this->green = std::max(this->green - cnst, 0);
	this->blue = std::max(this->blue - cnst, 0);
}

void Pixel::inverted_sub(int cnst){
	this->red = std::max(cnst - this->red, 0);
	this->green = std::max(cnst - this->green, 0);
	this->blue = std::max(cnst - this->blue, 0);
}

void Pixel::multiply(int cnst){
	this->red = std::min(this->red * cnst, 255);
	this->green = std::min(this->green * cnst, 255);
	this->blue = std::min(this->blue * cnst, 255);
}

void Pixel::divide(int cnst){
	if (cnst == 0) return;
	this->red /= cnst;
	this->green /= cnst;
	this->blue /= cnst;
}

void Pixel::inverted_divide(int cnst){
	if(this->red != 0)
		this->red = cnst / this->red;
	if(this->green != 0)
		this->green = cnst / this->green;
	if(this->blue != 0)
		this->blue = cnst / this->blue;
}

void Pixel::power(int cnst){
	this->red = std::min(int(std::pow(this->red, cnst)), 255);
	this->green = std::min(int(std::pow(this->green, cnst)), 255);
	this->blue = std::min(int(std::pow(this->blue, cnst)), 255);
}

void Pixel::log(){
	this->red = std::log(this->red + 1);
	this->green = std::log(this->green + 1);
	this->blue = std::log(this->blue + 1);
}

void Pixel::abs(){
	blue = std::abs(blue);
	green = std::abs(green);
	red = std::abs(red);
}

void Pixel::min(int cnst){
	this->red = std::min(this->red, cnst);
	this->green = std::min(this->green, cnst);
	this->blue = std::min(this->blue, cnst);
}

void Pixel::max(int cnst){
	this->red = std::max(this->red, cnst);
	this->green = std::max(this->green, cnst);
	this->blue = std::max(this->blue, cnst);
}

void Pixel::inversion(Pixel pixel){
	this->red = pixel.getRed() - this->red;
	this->green = pixel.getGreen() - this->green;
	this->blue = pixel.getBlue() - this->blue;
}

void Pixel::grayscale(){
	int tmp_red = this->red; int tmp_green = this->green; int tmp_blue = this->blue;
	this->red = (tmp_red + tmp_green + tmp_blue) / 3;
	this->green = (tmp_red + tmp_green + tmp_blue) / 3;
	this->blue = (tmp_red + tmp_green + tmp_blue) / 3;
}

void Pixel::filter(cv::Mat& filter){

}

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

class Pixel {
private:
	int red, green, blue;

public:
	Pixel(int r, int g, int b) {
		this->red = r;
		this->green = g;
		this->blue = b;
	}

	int getRed() {
		return red;
	}

	int getGreen() {
		return green;
	}

	int getBlue() {
		return blue;
	}

	void setRed(int red) {
		this->red = red;
	}

	void setGreen(int green) {
		this->green = green;
	}

	void setBlue(int blue) {
		this->blue = blue;
	}

	void add(int cnst);
	void sub(int cnst);
	void inverted_sub(int cnst);
	void multiply(int cnst);
	void divide(int cnst);
	void inverted_divide(int cnst);
	void power(int cnst);
	void log();
	void abs();
	void min(int cnst);
	void max(int cnst);
	
	void inversion();
	void grayscale();
	void filter(cv::Mat& filter);
};


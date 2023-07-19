#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <intrin.h>
#include "Pixel.h"


using namespace cv;

int get_cache_size() {
    int regs[4] = { 0 };
    __cpuid(regs, 0x80000006);
    int cacheLineSize = (regs[2] & 0xFF);
    printf("Cache Line Size: %d bytes\n", cacheLineSize);

    return cacheLineSize;
}

Pixel find_max(int rows, int cols, cv::Mat img) {
    int max_red = -1, max_green = -1, max_blue = -1;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
            Pixel pixel1(pixel[2], pixel[1], pixel[0]);

            if (pixel1.getRed() > max_red && pixel1.getGreen() > max_green && pixel1.getBlue() > max_blue) {
                max_red = pixel1.getRed();
                max_green = pixel1.getGreen();
                max_blue = pixel1.getBlue();
            }
        }
    }
    return Pixel(max_blue, max_green, max_red);
}




cv::Mat apply_filter(cv::Mat img, std::vector<std::vector<float>> kernel, int n, int rows, int cols) {
    cv::Mat tmp = img.clone();
    cv::Mat filteredImage = tmp.clone();
    int kernelHalfSize = n / 2;
    for (int i = -kernelHalfSize; i < rows - kernelHalfSize; i++) {
        for (int j = -kernelHalfSize; j < cols - kernelHalfSize; j++) {
            int sumR = 0, sumG = 0, sumB = 0;
            
            for (int k = 0; k < n; k++) {
                for (int l = 0; l < n; l++) {
                    if (i + k >= 0 && i + k < rows && j + l >= 0 && j + l < cols) {  //don't try to process pixels off the endge of the map
                        cv::Vec3b pixel_tmp = tmp.at<cv::Vec3b>(i + k, j + l);
                        sumR += pixel_tmp[2] * kernel[k][l];
                        sumG += pixel_tmp[1] * kernel[k][l];
                        sumB += pixel_tmp[0] * kernel[k][l];
                    }
                }
            }

            sumR = std::min(std::max(sumR, 0), 255);
            sumG = std::min(std::max(sumG, 0), 255);
            sumB = std::min(std::max(sumB, 0), 255);

            if(i >= 0 && j >= 0)
                filteredImage.at<cv::Vec3b>(i, j) = cv::Vec3b(sumB, sumG, sumR);

        }
    }
    return filteredImage;
}

cv::Mat apply_filter_optimized(cv::Mat img, std::vector<std::vector<float>> kernel, int n, int rows, int cols) {
    cv::Mat tmp = img.clone();
    cv::Mat filteredImage = tmp.clone();
    int kernelHalfSize = n / 2;
    int cache_line_size = get_cache_size();
    int blockSize = cache_line_size / sizeof(cv::Vec3b);
    blockSize = std::min(blockSize, n);

    for (int i = -kernelHalfSize; i < rows - kernelHalfSize; i += blockSize) {
        for (int j = -kernelHalfSize; j < cols - kernelHalfSize; j += blockSize) {
            for (int bi = i; bi < i + blockSize && bi < rows - kernelHalfSize; bi++) {
                for (int bj = j; bj < j + blockSize && bj < cols - kernelHalfSize; bj++) {
                    int sumR = 0, sumG = 0, sumB = 0;

                    for (int k = 0; k < n; k++) {
                        for (int l = 0; l < n; l++) {
                            if (bi + k >= 0 && bi + k < rows && bj + l >= 0 && bj + l < cols) {
                                cv::Vec3b pixel_tmp = tmp.at<cv::Vec3b>(bi + k, bj + l);
                                sumR += pixel_tmp[2] * kernel[k][l];
                                sumG += pixel_tmp[1] * kernel[k][l];
                                sumB += pixel_tmp[0] * kernel[k][l];
                            }
                        }
                    }

                    sumR = std::min(std::max(sumR, 0), 255);
                    sumG = std::min(std::max(sumG, 0), 255);
                    sumB = std::min(std::max(sumB, 0), 255);

                    if (bi >= 0 && bj >= 0)
                        filteredImage.at<cv::Vec3b>(bi, bj) = cv::Vec3b(sumB, sumG, sumR);
                }
            }
        }
    }
    return filteredImage;
}

int main() {
    //std::cout << "cache line size: " << get_cache_size();
	cv::Mat img = cv::imread("C:/Users/anjci/OneDrive/Documents/home_page_monet.jpg");
    //cv::Mat kernel = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    //cv::Mat kernel;
    int n;

    std::cout << "Insert kernel filter size: ";
    std::cin >> n;

    std::vector<std::vector<float>> kernel(n, std::vector<float>(n));

    std::cout << "Enter the elements of the matrix:\n";

    // Read each element of the matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << "Enter element at position (" << i << ", " << j << "): ";
            std::cin >> kernel[i][j];
        }
    }

	if (img.empty())
	{
		std::cout << "Failed to load the image." << std::endl;
		return -1;
	}
    int rows = img.rows;
    int cols = img.cols;

    // Create a new image to store the result
    cv::Mat result(img.size(), img.type());

    Pixel max_pixel = find_max(rows, cols, img);

    cv::Mat filtered_img = apply_filter(img, kernel, n, rows, cols);
    cv::imshow("Filtered", filtered_img);
    cv::waitKey(0);


    cv::Mat filtered_img_opt = apply_filter_optimized(img, kernel, n, rows, cols);


    cv::imshow("Filtered opt", filtered_img_opt);
    cv::waitKey(0);


    // Iterate over each pixel
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // Access pixel values at (i, j)
            cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);

            // Create Pixel objects
            Pixel pixel1(pixel[2], pixel[1], pixel[0]);
            //Pixel pixel2(50, 50, 50);  // Constant pixel value

            // Perform arithmetic operations on pixels
            Pixel resultPixel = pixel1;
            //resultPixel.add(50);          // Addition
            //resultPixel.sub(50);     // Subtraction
            //resultPixel.inverted_sub(150);
            //resultPixel.multiply(5);     // Multiplication
            //resultPixel.divide(4);       // Division
            //resultPixel.inverted_divide(255);
            //resultPixel.power(2);          // Power
            //resultPixel.log();          // Logarithm
            //resultPixel.abs();           // Absolute
            //resultPixel.min(135);      // Minimum
            //resultPixel.max(115);      // Maximum

            //resultPixel.inversion(max_pixel);
            resultPixel.grayscale();

            // Update the result image with the modified pixel values
            result.at<cv::Vec3b>(i, j) = cv::Vec3b(resultPixel.getBlue(), resultPixel.getGreen(), resultPixel.getRed());
        }
    }

    // Display the result image
    cv::imshow("Result", result);
	//namedWindow("First open cv app", WINDOW_AUTOSIZE);
	//cv::imshow("first opencv app", img);
	//cv::moveWindow("First open cv app", 0, 45);
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}
#pragma once

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
using namespace std;

class ResultImage {
private :
	Mat &result;
    int rows, cols;

public:
	ResultImage(Mat &image): result(image){
        this->rows = result.rows;
        this->cols = result.cols;
	}

    Mat get_result() {
        return result;
    }

	ResultImage& apply_add(int cnst) {

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                Vec3b pixel = result.at<cv::Vec3b>(i, j);

                // create pixel object
                //Pixel pixel1(pixel[2], pixel[1], pixel[0]);
                //Pixel resultPixel = pixel1;

                //resultPixel.add(cnst);

                int red = pixel[2] + cnst;
                int green = pixel[1] + cnst;
                int blue = pixel[0] + cnst;

                if (red > 255)
                    red = 255;
                if (green > 255)
                    green = 255;
                if (blue > 255)
                    blue = 255;

                result.at<Vec3b>(i, j) = Vec3b(blue, green, red);
            }
        }

        return *this;
	}


    ResultImage& apply_optimized_add(int cnst) {
        int stride = this->cols % 8; // if img not a multiple of 8
        Mat new_result(rows, cols, CV_8UC3);

        __m256i constant_vector = _mm256_set1_epi8(static_cast<char>(cnst));
        int i, j;

        for (i = 0; i < this->rows; i++) {
            for (j = 0; j < this->cols - stride; j += 8) {
                // gets 8 pixels starting from the address pixel_ptr
                Vec3b* pixel_ptr = &result.at<Vec3b>(i, j);
                Vec3b* result_pixel_ptr = &new_result.at<Vec3b>(i, j);

                __m256i result_pixels = _mm256_load_si256(reinterpret_cast<const __m256i*>(pixel_ptr));

                // adding those pixels with constants
                result_pixels = _mm256_adds_epu8(result_pixels, constant_vector);

                // storing that result 
                _mm256_store_si256(reinterpret_cast<__m256i*>(result_pixel_ptr), result_pixels);
            }
        }
        // for the leftovers
        for (; i < this->rows; i++) {
            for (j = 0; j < this->cols; j++) {
                cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
                int blue = pixel[0];
                int green = pixel[1];
                int red = pixel[2];

                // perform arithmetic operations on the pixel values
                int new_blue = blue + cnst;
                int new_green = green + cnst;
                int new_red = red + cnst;

                if (new_blue > 255) new_blue = 255;
                if (new_green > 255) new_green = 255;
                if (new_red > 255) new_red = 255;

                // Update the pixel values
                new_result.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(new_blue), uchar(new_green), uchar(new_red));
            }
        }

        this->result = new_result;
        this->cols = new_result.cols;
        this->rows = new_result.rows;

        return *this;
    }

    ResultImage& apply_sub(int cnst) {
        // iterate over each pixel
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                Vec3b pixel = result.at<cv::Vec3b>(i, j);

                // create Pixel objects
                Pixel pixel1(pixel[2], pixel[1], pixel[0]);
                Pixel resultPixel = pixel1;
                resultPixel.sub(cnst);   

                result.at<Vec3b>(i, j) = Vec3b(resultPixel.getBlue(), resultPixel.getGreen(), resultPixel.getRed());
            }
        }

        return *this;
    }

    ResultImage& apply_optimized_sub(int cnst) {
        int stride = this->cols % 8; // if img not a multiple of 8
        Mat new_result(rows, cols, CV_8UC3);

        __m256i constant_vector = _mm256_set1_epi8(static_cast<char>(cnst));
        int i, j;

        for (i = 0; i < this->rows; i++) {
            for (j = 0; j < this->cols - stride; j += 8) {
                // Gets 8 pixels starting from the address pixel_ptr
                Vec3b* pixel_ptr = &result.at<Vec3b>(i, j);
                Vec3b* result_pixel_ptr = &new_result.at<Vec3b>(i, j);

                __m256i result_pixels = _mm256_load_si256(reinterpret_cast<const __m256i*>(pixel_ptr));

                // Adding those pixels with constants
                result_pixels = _mm256_subs_epu8(result_pixels, constant_vector);


                // Storing that result 
                _mm256_store_si256(reinterpret_cast<__m256i*>(result_pixel_ptr), result_pixels);
            }
        }

        for (; i < this->rows; i++) {
            for (j = 0; j < this->cols; j++) {
                cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
                int blue = pixel[0];
                int green = pixel[1];
                int red = pixel[2];

                // perform arithmetic operations on the pixel values
                int newBlue = blue - cnst;
                int newGreen = green - cnst;
                int newRed = red - cnst;

                if (newBlue < 0) newBlue = 0;
                if (newGreen < 0) newGreen = 0;
                if (newRed < 0) newRed = 0;

                // update the pixel values
                new_result.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(newBlue), uchar(newGreen), uchar(newRed));
            }
        }

        this->result = new_result;
        this->cols = new_result.cols;
        this->rows = new_result.rows;

        return *this;
    }

    ResultImage& apply_inverted_sub(int cnst) {
        // Iterate over each pixel
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                Vec3b pixel = result.at<cv::Vec3b>(i, j);

                // Create Pixel objects
                Pixel pixel1(pixel[2], pixel[1], pixel[0]);
                Pixel resultPixel = pixel1;
                resultPixel.inverted_sub(cnst);
                //resultPixel = apply_add(resultPixel);    

                result.at<Vec3b>(i, j) = Vec3b(resultPixel.getBlue(), resultPixel.getGreen(), resultPixel.getRed());
            }
        }

        return *this;
    }

    ResultImage& apply_optimized_inverted_sub(int cnst) {
        int stride = this->cols % 8; // if img not a multiple of 8
        Mat new_result(rows, cols, CV_8UC3);

        __m256i constant_vector = _mm256_set1_epi8(static_cast<char>(cnst));
        int i, j;

        for (i = 0; i < this->rows; i++) {
            for (j = 0; j < this->cols - stride; j += 8) {
                // Gets 8 pixels starting from the address pixel_ptr
                Vec3b* pixel_ptr = &result.at<Vec3b>(i, j);
                Vec3b* result_pixel_ptr = &new_result.at<Vec3b>(i, j);

                __m256i result_pixels = _mm256_load_si256(reinterpret_cast<const __m256i*>(pixel_ptr));

                // Adding those pixels with constants
                result_pixels = _mm256_subs_epu8(constant_vector, result_pixels);


                // Storing that result 
                _mm256_store_si256(reinterpret_cast<__m256i*>(result_pixel_ptr), result_pixels);
            }
        }

        for (; i < this->rows; i++) {
            for (j = 0; j < this->cols; j++) {
                cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
                int blue = pixel[0];
                int green = pixel[1];
                int red = pixel[2];

                // Perform arithmetic operations on the pixel values
                int newBlue = cnst - blue;
                int newGreen = cnst - green;
                int newRed = cnst - red;

                if (newBlue < 0) newBlue = 0;
                if (newGreen < 0) newGreen = 0;
                if (newRed < 0) newRed = 0;

                // Update the pixel values
                new_result.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(newBlue), uchar(newGreen), uchar(newRed));
            }
        }

        this->result = new_result;
        this->cols = new_result.cols;
        this->rows = new_result.rows;

        return *this;
    }

    ResultImage& apply_mul(int cnst) {
        // Iterate over each pixel
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                Vec3b pixel = result.at<cv::Vec3b>(i, j);

                // Create Pixel objects
                Pixel pixel1(pixel[2], pixel[1], pixel[0]);
                Pixel resultPixel = pixel1;
                resultPixel.multiply(cnst);
                
                result.at<Vec3b>(i, j) = Vec3b(resultPixel.getBlue(), resultPixel.getGreen(), resultPixel.getRed());
            }
        }

        return *this;
    }

    ResultImage& apply_optimized_mul(int cnst) {
        int stride = this->cols % 8; // if img not a multiple of 8
        Mat new_result(rows, cols, CV_8UC3);

        __m256i constant_vector = _mm256_set1_epi8(static_cast<char>(cnst));
        __m256i constant_vector_expanded = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(constant_vector)); // expands the constant_vector from 8 bits to 16
        
        int i, j;

        for (i = 0; i < this->rows; i++) {
            for (j = 0; j < this->cols - stride; j += 8) {
                // Gets 8 pixels starting from the address pixel_ptr
                Vec3b* pixel_ptr = &result.at<Vec3b>(i, j);
                Vec3b* result_pixel_ptr = &new_result.at<Vec3b>(i, j);

                __m256i pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result.at<Vec3b>(i, j)));

                // initialize result_pixels with zeros
                __m256i result_pixels = _mm256_setzero_si256();

                // add the loaded pixels repeatedly to simulate multiplication
                for (int k = 0; k < cnst; k++) {
                    result_pixels = _mm256_adds_epu8(result_pixels, pixels);
                }
               
                // Storing that result 
                _mm256_store_si256(reinterpret_cast<__m256i*>(result_pixel_ptr), result_pixels);
            }
        }

        for (; i < this->rows; i++) {
            for (j = 0; j < this->cols; j++) {
                cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
                int blue = pixel[0];
                int green = pixel[1];
                int red = pixel[2];

                // Perform arithmetic operations on the pixel values
                int newBlue = blue * cnst;
                int newGreen = green * cnst;
                int newRed = red * cnst;

                if (newBlue > 255) newBlue = 255;
                if (newGreen > 0) newGreen = 255;
                if (newRed > 255) newRed = 255;

                // Update the pixel values
                new_result.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(newBlue), uchar(newGreen), uchar(newRed));
            }
        }


        this->result = new_result;
        this->cols = new_result.cols;
        this->rows = new_result.rows;

        return *this;
    }

    ResultImage& apply_div(int cnst) {
        // Iterate over each pixel
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                Vec3b pixel = result.at<cv::Vec3b>(i, j);

                // Create Pixel objects
                Pixel pixel1(pixel[2], pixel[1], pixel[0]);
                Pixel resultPixel = pixel1;
                resultPixel.divide(cnst);  

                result.at<Vec3b>(i, j) = Vec3b(resultPixel.getBlue(), resultPixel.getGreen(), resultPixel.getRed());
            }
        }

        return *this;
    }

    ResultImage& apply_optimized_div(int cnst) {
        if (cnst <= 0)
            return *this;
        int stride = this->cols % 8; // if img not a multiple of 8
        Mat new_result(rows, cols, CV_8UC3);
        __m256i constant_vector = _mm256_set1_epi8(static_cast<char>(cnst));
        int i, j;

        for (i = 0; i < this->rows; i++) {
            for (j = 0; j < this->cols - stride; j += 8) {
                // Gets 8 pixels starting from the address pixel_ptr
                Vec3b* pixel_ptr = &result.at<Vec3b>(i, j);
                Vec3b* result_pixel_ptr = &new_result.at<Vec3b>(i, j);

                __m256i result_pixels = _mm256_load_si256(reinterpret_cast<const __m256i*>(pixel_ptr));

                // Adding those pixels with constants
                result_pixels = _mm256_div_epu8(result_pixels, constant_vector);


                // Storing that result 
                _mm256_store_si256(reinterpret_cast<__m256i*>(result_pixel_ptr), result_pixels);
            }
        }

        for (; i < this->rows; i++) {
            for (j = 0; j < this->cols; j++) {
                cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
                int blue = pixel[0];
                int green = pixel[1];
                int red = pixel[2];

                // Perform arithmetic operations on the pixel values
                int newBlue = blue / cnst;
                int newGreen = green / cnst;
                int newRed = red / cnst;

                if (newBlue < 0) newBlue = 0;
                if (newGreen < 0) newGreen = 0;
                if (newRed < 0) newRed = 0;

                // Update the pixel values
                new_result.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(newBlue), uchar(newGreen), uchar(newRed));
            }
        }


        this->result = new_result;
        this->cols = new_result.cols;
        this->rows = new_result.rows;

        return *this;
    }

    ResultImage& apply_inverted_div(int cnst) {
        // Iterate over each pixel
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                Vec3b pixel = result.at<cv::Vec3b>(i, j);

                // Create Pixel objects
                Pixel pixel1(pixel[2], pixel[1], pixel[0]);
                Pixel resultPixel = pixel1;
                resultPixel.inverted_divide(cnst);   

                result.at<Vec3b>(i, j) = Vec3b(resultPixel.getBlue(), resultPixel.getGreen(), resultPixel.getRed());
            }
        }

        return *this;
    }

    ResultImage& apply_optimized_inverted_div(int cnst) {
        
        int stride = this->cols % 8; // if img not a multiple of 8
        Mat new_result(rows, cols, CV_8UC3);
        __m256i constant_vector = _mm256_set1_epi8(static_cast<char>(cnst));
        //__m256i ones = _mm256_set1_epi8(static_cast<char>(1));
        __m256i ones = _mm256_set1_epi8(1);
        int i, j;

        for (i = 0; i < this->rows; i++) {
            for (j = 0; j < this->cols - stride; j += 8) {
                // Gets 8 pixels starting from the address pixel_ptr
                Vec3b* pixel_ptr = &result.at<Vec3b>(i, j);
                Vec3b* result_pixel_ptr = &new_result.at<Vec3b>(i, j);

                __m256i result_pixels = _mm256_load_si256(reinterpret_cast<const __m256i*>(pixel_ptr));

                // checks for zero values in result_pixels
                __m256i zero_check = _mm256_cmpeq_epi8(result_pixels, _mm256_setzero_si256());
                // if there is using and operation we add one to that zero element in result_pixels
                result_pixels = _mm256_adds_epi8(result_pixels, _mm256_and_si256(zero_check, ones));

                result_pixels = _mm256_div_epu8(constant_vector, result_pixels);

                _mm256_store_si256(reinterpret_cast<__m256i*>(result_pixel_ptr), result_pixels);
            }
        }

        for (; i < this->rows; i++) {
            for (j = 0; j < this->cols; j++) {
                cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
                int blue = pixel[0];
                int green = pixel[1];
                int red = pixel[2];

                // Perform arithmetic operations on the pixel values
                int newBlue = 0, newGreen = 0, newRed = 0;
                if (blue != 0)
                    newBlue = cnst / blue;
                if (blue != 0)
                    newGreen = cnst / green;
                if (red != 0)
                    newRed = cnst / red;

                if (newBlue < 0) newBlue = 0;
                if (newGreen < 0) newGreen = 0;
                if (newRed < 0) newRed = 0;

                // Update the pixel values
                new_result.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(newBlue), uchar(newGreen), uchar(newRed));
            }
        }


        this->result = new_result;
        this->cols = new_result.cols;
        this->rows = new_result.rows;

        return *this;
    }

    ResultImage& apply_pow(int cnst) {
        // Iterate over each pixel
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                Vec3b pixel = result.at<cv::Vec3b>(i, j);

                // Create Pixel objects
                Pixel pixel1(pixel[2], pixel[1], pixel[0]);
                Pixel resultPixel = pixel1;
                resultPixel.power(cnst);    

                result.at<Vec3b>(i, j) = Vec3b(resultPixel.getBlue(), resultPixel.getGreen(), resultPixel.getRed());
            }
        }

        return *this;
    }

    ResultImage& apply_optimized_pow(int cnst) {
        if (cnst == 0)
            return *this;
        int stride = this->cols % 8; // if img not a multiple of 8
        Mat new_result(rows, cols, CV_8UC3);

        __m256i constant_vector = _mm256_set1_epi8(static_cast<char>(cnst));
        __m256i zeros = _mm256_set1_epi8(static_cast<char>(0));

        int n = cnst;

        int i, j;


        for (i = 0; i < this->rows; i++) {
            for (j = 0; j < this->cols - stride; j += 8) {
                // Gets 8 pixels starting from the address pixel_ptr
                Vec3b* pixel_ptr = &result.at<Vec3b>(i, j);
                Vec3b* result_pixel_ptr = &new_result.at<Vec3b>(i, j);

                __m256i result_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result.at<Vec3b>(i, j)));
                __m256i answer = result_pixels;
                __m256i increment = result_pixels;

                int k = 0;

                for (int x = 0; x < cnst; x++)
                {
                    k = 0;
                    for (int y = 0; y < result_pixels.m256i_u8[k++]; y++) {
                        if (k == 7)
                            break;
                        answer = _mm256_adds_epu8(answer, increment);
                    }
                    increment = answer;
                }

                result_pixels = answer;

                /*for (int k = 0; k < cnst * cnst; k++) {
                    result_pixels = _mm256_adds_epu8(result_pixels, result_pixels);
                }*/
                // Storing that result 
                _mm256_store_si256(reinterpret_cast<__m256i *>(result_pixel_ptr), result_pixels);
            }
        }

        for (; i < this->rows; i++) {
            for (j = 0; j < this->cols; j++) {
                cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
                int blue = pixel[0];
                int green = pixel[1];
                int red = pixel[2];

                // Perform arithmetic operations on the pixel values
                int newBlue = std::pow(blue, cnst);
                int newGreen = std::pow(green, cnst);
                int newRed = std::pow(red, cnst);

                if (newBlue > 255) newBlue = 255;
                if (newGreen > 255) newGreen = 255;
                if (newRed > 255) newRed = 255;

                // Update the pixel values
                new_result.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(newBlue), uchar(newGreen), uchar(newRed));
            }
        }


        this->result = new_result;
        this->cols = new_result.cols;
        this->rows = new_result.rows;

        return *this;
    }

    ResultImage& apply_log() {
        // Iterate over each pixel
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                Vec3b pixel = result.at<cv::Vec3b>(i, j);

                // Create Pixel objects
                Pixel pixel1(pixel[2], pixel[1], pixel[0]);
                Pixel resultPixel = pixel1;
                resultPixel.log();
                
                result.at<Vec3b>(i, j) = Vec3b(resultPixel.getBlue(), resultPixel.getGreen(), resultPixel.getRed());
            }
        }

        return *this;
    }


    ResultImage& apply_optimized_log() {
        int stride = this->cols % 8; // if img not a multiple of 8
        Mat new_result(rows, cols, CV_8UC3);
        __m256 max_vector = _mm256_set1_ps(255);
        __m256 ones = _mm256_set1_ps(1);
        __m256 p_black = _mm256_set1_ps(245);
        __m256 p_white = _mm256_set1_ps(254);
        //__m256 constant_vector = _mm256_set1_ps(constant);

        int i, j;

        // for rgb values
        int R[8], G[8], B[8];


        for (i = 0; i < this->rows; i++) {
            for (j = 0; j < this->cols - stride; j += 8) {
                // Gets 8 pixels starting from the address pixel_ptr
                Vec3b* pixel_ptr = &result.at<Vec3b>(i, j);
                Vec3b* result_pixel_ptr = &new_result.at<Vec3b>(i, j);

                // gets rgb values
                for (int k = 0; k < 8; k++) {
                    register int x = 0;
                    Vec3b pixels = result.at<Vec3b>(i, j + k);

                    B[k] = pixels[x++];
                    G[k] = pixels[x++];
                    R[k] = pixels[x++];

                }

                // sets the vectors
                __m256 vR = _mm256_setr_ps(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7]);
                __m256 vG = _mm256_setr_ps(G[0], G[1], G[2], G[3], G[4], G[5], G[6], G[7]);
                __m256 vB = _mm256_setr_ps(B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7]);

                __m256 log_red = _mm256_log_ps(vR);
                __m256 log_green = _mm256_log_ps(vG);
                __m256 log_blue = _mm256_log_ps(vB);
                
                // this is the formula i've been using
                // 	this->red = 255 * ((std::log(this->red - p_black)) / (std::log(p_white - p_black)));

               /* __m256 log_first_red = _mm256_log_ps(_mm256_sub_ps(vR, p_black));
                __m256 log_first_green = _mm256_log_ps(_mm256_sub_ps(vG, p_black));
                __m256 log_first_blue = _mm256_log_ps(_mm256_sub_ps(vB, p_black));

                __m256 log_second = _mm256_log_ps(_mm256_sub_ps(p_white, p_black));

                __m256 red_divided = _mm256_div_ps(log_first_red, log_second);
                __m256 green_divided = _mm256_div_ps(log_first_green, log_second);
                __m256 blue_divided = _mm256_div_ps(log_first_blue, log_second);

                __m256 sum_pixels_red = _mm256_setzero_ps();
                __m256 sum_pixels_green = _mm256_setzero_ps();
                __m256 sum_pixels_blue = _mm256_setzero_ps();*/

                // add the loaded pixels repeatedly to simulate multiplication
                /*for (int k = 0; k < 255; k++) {
                    red_divided = _mm256_add_ps(sum_pixels_red, red_divided);
                    blue_divided = _mm256_add_ps(sum_pixels_blue, blue_divided);
                    green_divided = _mm256_add_ps(sum_pixels_green, green_divided);
                }*/
                
                // converts to int
                __m256i red_int = _mm256_cvtps_epi32(log_red);
                __m256i green_int = _mm256_cvtps_epi32(log_green);
                __m256i blue_int = _mm256_cvtps_epi32(log_blue);

                for (int k = 0; k < 8; k++)
                {
                    new_result.at<cv::Vec3b>(i, j + k) = cv::Vec3b(blue_int.m256i_i32[k], green_int.m256i_i32[k], red_int.m256i_i32[k]);
                }
}
        }

        for (; i < this->rows; i++) {
            for (j = 0; j < this->cols; j++) {
                cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
                int blue = pixel[0];
                int green = pixel[1];
                int red = pixel[2];

                // Perform arithmetic operations on the pixel values
                int newBlue = blue;
                int newGreen = green;
                int newRed = red;

                if (newBlue < 0) newBlue = 0;
                if (newGreen < 0) newGreen = 0;
                if (newRed < 0) newRed = 0;

                // Update the pixel values
                new_result.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(newBlue), uchar(newGreen), uchar(newRed));
            }
        }

        this->result = new_result;
        this->cols = new_result.cols;
        this->rows = new_result.rows;

        return *this;
    }

    ResultImage& apply_abs() {
        // Iterate over each pixel
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                Vec3b pixel = result.at<cv::Vec3b>(i, j);

                // Create Pixel objects
                Pixel pixel1(pixel[2], pixel[1], pixel[0]);
                Pixel resultPixel = pixel1;
                resultPixel.abs();
                //resultPixel = apply_add(resultPixel);    

                result.at<Vec3b>(i, j) = Vec3b(resultPixel.getBlue(), resultPixel.getGreen(), resultPixel.getRed());
            }
        }

        return *this;
    }

    ResultImage& apply_min(int cnst) {
        // Iterate over each pixel
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                Vec3b pixel = result.at<cv::Vec3b>(i, j);

                // Create Pixel objects
                Pixel pixel1(pixel[2], pixel[1], pixel[0]);
                Pixel resultPixel = pixel1;
                resultPixel.min(cnst);
                result.at<Vec3b>(i, j) = Vec3b(resultPixel.getBlue(), resultPixel.getGreen(), resultPixel.getRed());
            }
        }

        return *this;
    }


    ResultImage& apply_optimized_min(int cnst) {
        int stride = this->cols % 8; // if img not a multiple of 8
        Mat new_result(rows, cols, CV_8UC3);

        __m256i constant_vector = _mm256_set1_epi8(static_cast<char>(cnst));
        
        int i, j;


        for (i = 0; i < this->rows; i++) {
            for (j = 0; j < this->cols - stride; j += 8) {
                // Gets 8 pixels starting from the address pixel_ptr
                Vec3b* pixel_ptr = &result.at<Vec3b>(i, j);
                Vec3b* result_pixel_ptr = &new_result.at<Vec3b>(i, j);

                __m256i result_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result.at<Vec3b>(i, j)));
                //_mm256_pow_ps(result_pixels, constant_vector);
                result_pixels = _mm256_min_epu8(result_pixels, constant_vector);
                // Storing that result 
                _mm256_store_si256(reinterpret_cast<__m256i*>(result_pixel_ptr), result_pixels);
            }
        }

        for (; i < this->rows; i++) {
            for (j = 0; j < this->cols; j++) {
                cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
                int blue = pixel[0];
                int green = pixel[1];
                int red = pixel[2];

                // Perform arithmetic operations on the pixel values
                int newBlue = std::min(blue, cnst);
                int newGreen = std::min(green, cnst);
                int newRed = std::min(red, cnst);

                if (newBlue < 0) newBlue = 0;
                if (newGreen < 0) newGreen = 0;
                if (newRed < 0) newRed = 0;

                // Update the pixel values
                new_result.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(newBlue), uchar(newGreen), uchar(newRed));
            }
        }


        this->result = new_result;
        this->cols = new_result.cols;
        this->rows = new_result.rows;

        return *this;
    }

    ResultImage& apply_max(int cnst) {
        // Iterate over each pixel
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                Vec3b pixel = result.at<cv::Vec3b>(i, j);

                // Create Pixel objects
                Pixel pixel1(pixel[2], pixel[1], pixel[0]);
                Pixel resultPixel = pixel1;
                resultPixel.max(cnst);  

                result.at<Vec3b>(i, j) = Vec3b(resultPixel.getBlue(), resultPixel.getGreen(), resultPixel.getRed());
            }
        }

        return *this;
    }

    ResultImage& apply_optimized_max(int cnst) {
        int stride = this->cols % 8; // if img not a multiple of 8
        Mat new_result(rows, cols, CV_8UC3);
        __m256i constant_vector = _mm256_set1_epi8(static_cast<char>(cnst));
        
        int i, j;


        for (i = 0; i < this->rows; i++) {
            for (j = 0; j < this->cols - stride; j += 8) {
                // Gets 8 pixels starting from the address pixel_ptr
                Vec3b* pixel_ptr = &result.at<Vec3b>(i, j);
                Vec3b* result_pixel_ptr = &new_result.at<Vec3b>(i, j);

                __m256i result_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result.at<Vec3b>(i, j)));
                //_mm256_pow_ps(result_pixels, constant_vector);
                result_pixels = _mm256_max_epu8(result_pixels, constant_vector);
                // Storing that result 
                _mm256_store_si256(reinterpret_cast<__m256i*>(result_pixel_ptr), result_pixels);
            }
        }

        for (; i < this->rows; i++) {
            for (j = 0; j < this->cols; j++) {
                cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
                int blue = pixel[0];
                int green = pixel[1];
                int red = pixel[2];

                // Perform arithmetic operations on the pixel values
                int newBlue = std::max(blue, cnst);
                int newGreen = std::max(green, cnst);
                int newRed = std::max(red, cnst);

                if (newBlue < 0) newBlue = 0;
                if (newGreen < 0) newGreen = 0;
                if (newRed < 0) newRed = 0;

                // Update the pixel values
                new_result.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(newBlue), uchar(newGreen), uchar(newRed));
            }
        }


        this->result = new_result;
        this->cols = new_result.cols;
        this->rows = new_result.rows;

        return *this;
    }


    ResultImage& apply_inversion() {
        // Iterate over each pixel
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                Vec3b pixel = result.at<cv::Vec3b>(i, j);

                // Create Pixel objects
                Pixel pixel1(pixel[2], pixel[1], pixel[0]);
                Pixel resultPixel = pixel1;
                resultPixel.inversion();    

                result.at<Vec3b>(i, j) = Vec3b(resultPixel.getBlue(), resultPixel.getGreen(), resultPixel.getRed());
            }
        }

        return *this;
    }


    ResultImage& apply_optimized_inversion() {
        int constant = 255;
        int stride = this->cols % 8; // if img not a multiple of 8
        Mat new_result(rows, cols, CV_8UC3);

        __m256i constant_vector = _mm256_set1_epi8(static_cast<char>(constant));

        int i, j;


        for (i = 0; i < this->rows; i++) {
            for (j = 0; j < this->cols - stride; j += 8) {
                // Gets 8 pixels starting from the address pixel_ptr
                Vec3b* pixel_ptr = &result.at<Vec3b>(i, j);
                Vec3b* result_pixel_ptr = &new_result.at<Vec3b>(i, j);

                __m256i result_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&result.at<Vec3b>(i, j)));
                result_pixels = _mm256_subs_epu8(constant_vector, result_pixels);
                // Storing that result 
                _mm256_store_si256(reinterpret_cast<__m256i*>(result_pixel_ptr), result_pixels);
            }
        }

        for (; i < this->rows; i++) {
            for (j = 0; j < this->cols; j++) {
                cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
                int blue = pixel[0];
                int green = pixel[1];
                int red = pixel[2];

                // Perform arithmetic operations on the pixel values
                int newBlue = constant - blue;
                int newGreen = constant - green;
                int newRed = constant - red;

                if (newBlue < 0) newBlue = 0;
                if (newGreen < 0) newGreen = 0;
                if (newRed < 0) newRed = 0;

                // Update the pixel values
                new_result.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(newBlue), uchar(newGreen), uchar(newRed));
            }
        }


        this->result = new_result;
        this->cols = new_result.cols;
        this->rows = new_result.rows;

        return *this;
    }

    ResultImage& apply_grayscale() {
        // Iterate over each pixel
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {

                Vec3b pixel = result.at<cv::Vec3b>(i, j);

                // Create Pixel objects
                Pixel pixel1(pixel[2], pixel[1], pixel[0]);
                Pixel resultPixel = pixel1;
                resultPixel.grayscale();   

                result.at<Vec3b>(i, j) = Vec3b(resultPixel.getBlue(), resultPixel.getGreen(), resultPixel.getRed());
            }
        }

        return *this;
    }

    ResultImage& apply_optimized_grayscale() {
        int constant = 3;
        int stride = this->cols % 8; // if img not a multiple of 8
        Mat new_result(rows, cols, CV_8UC3);

        __m256i constant_vector = _mm256_set1_epi8(static_cast<char>(constant));
        int i, j;
        int res = 0;

        
        uint8_t R[8], G[8], B[8], A[8];


        __m256 vCoefR = _mm256_set1_ps(0.299);
        __m256 vCoefG = _mm256_set1_ps(0.587);
        __m256 vCoefB = _mm256_set1_ps(0.114);

        for (i = 0; i < this->rows; i++) {
            for (j = 0; j < this->cols - stride; j += 8) {
               
                for (int k = 0; k < 8; k++) {
                    register int x = 0;
                    Vec3b pixels = result.at<Vec3b>(i, j + k);

                    B[k] = pixels[x++];
                    G[k] = pixels[x++];
                    R[k] = pixels[x++];
                    A[k] = pixels[x++];
                }

                
                __m256 vR = _mm256_setr_ps(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7]);
                __m256 sum = _mm256_mul_ps(vR, vCoefR);

                
                __m256 vG = _mm256_setr_ps(G[0], G[1], G[2], G[3], G[4], G[5], G[6], G[7]);
                sum = _mm256_fmadd_ps(vG, vCoefG, sum);


                
                __m256 vB = _mm256_setr_ps(B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7]);
                sum = _mm256_fmadd_ps(vB, vCoefB, sum);


                
                __m256i sumI = _mm256_cvtps_epi32(sum);

                
                for (int k = 0; k < 8; k++) {
                    uint16_t _gray = sumI.m256i_u8[k * 4];
                    
                    Vec3b gray_pixel(_gray, _gray, _gray);

                    new_result.at<Vec3b>(i, j + k) = gray_pixel;
                    
                }

                
             }
        }

        
        for (; i < this->rows; i++) {
            for (j = 0; j < this->cols; j++) {
                cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
                int blue = pixel[0];
                int green = pixel[1];
                int red = pixel[2];

                // Perform arithmetic operations on the pixel values
                int newBlue = blue - constant;
                int newGreen = green - constant;
                int newRed = red - constant;

                if (newBlue < 0) newBlue = 0;
                if (newGreen < 0) newGreen = 0;
                if (newRed < 0) newRed = 0;

                // Update the pixel values
                new_result.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(newBlue), uchar(newGreen), uchar(newRed));
            }
        }


        this->result = new_result;
        this->cols = new_result.cols;
        this->rows = new_result.rows;

        return *this;
    }

    ResultImage& apply_filter(std::vector<std::vector<float>> kernel, int n) {
        // Iterate over each pixel
       // Mat original_image = result.clone();
        //Mat original_image = tmp.clone();
        Mat filtered_image(rows, cols, CV_8UC3, Scalar(0, 0, 0));

        int kernel_half_size = n / 2;

        // iterating through the picture
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int sumR = 0, sumG = 0, sumB = 0;

                // iterating through kernel
                for (int k = 0; k < n; k++) {
                    for (int l = 0; l < n; l++) {
                        if (i + k - kernel_half_size >= 0 && i + k - kernel_half_size < rows && j + l - kernel_half_size >= 0 && j + l - kernel_half_size < cols) {
                            // cv::Vec3b pixel_tmp = tmp.at<cv::Vec3b>(i + k - kernel_half_size, j + l - kernel_half_size);
                            Vec3b pixel_tmp = result.at<Vec3b>(i + k - kernel_half_size, j + l - kernel_half_size);

                            sumR += pixel_tmp[2] * kernel[k][l];
                            sumG += pixel_tmp[1] * kernel[k][l];
                            sumB += pixel_tmp[0] * kernel[k][l];
                            // fmadd
                        }
                    }
                }

                if (i >= 0 && j >= 0)
                    filtered_image.at<Vec3b>(i, j) = Vec3b(sumB, sumG, sumR);
            }
        }

        this->result = filtered_image;
        this->cols = filtered_image.cols;
        this->rows = filtered_image.rows;

        return *this;
    }


   ResultImage& apply_optimized_filter(std::vector<std::vector<float>> kernel, int n) {
        int constant = 3;
        int stride = this->cols % 8; // if img not a multiple of 8
        Mat new_result(rows, cols, CV_8UC3);

        int kernel_half_size = n / 2;
       // __m256i kernel_half_size = _mm256_set1_epi8(static_cast<char>(n / 2));
        int i, j;
        int res = 0;


        uint8_t R[8], G[8], B[8], A[8];

        for (i = 0; i < this->rows; i++) {
            for (j = 0; j < this->cols - stride; j += 8) {
                int y = 0;
                // iterating through kernel and getting pixels
                for (int k = 0; k < n; k++) {
                    //if (y == 8) break;

                    for (int l = 0; l < n; l++) {
                        
                        register int x = 0;
                        if (i + k - kernel_half_size >= 0 && i + k - kernel_half_size < rows && j + l - kernel_half_size >= 0 && j + l - kernel_half_size < cols) {
                            //cout <<"k " << k << " l " << l << " curr " << (k * n + l) << endl;
                            Vec3b pixels = result.at<Vec3b>(i + k - kernel_half_size, j + l - kernel_half_size);
                            if ((k * n + l) != 8) {
                                B[k * n + l] = pixels[x++];
                                G[k * n + l] = pixels[x++];
                                R[k * n + l] = pixels[x++];
                            }
                            //cout << "B " << B[k * n + l] << " G " << G[k * n + l] << " R " << R[k * n + l] << endl;
                        }

                    }
                }


                __m256 vR = _mm256_setr_ps(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7]);

                __m256 vG = _mm256_setr_ps(G[0], G[1], G[2], G[3], G[4], G[5], G[6], G[7]);

                __m256 vB = _mm256_setr_ps(B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7]);


                __m256 sumR = _mm256_setzero_ps();
                __m256 sumG = _mm256_setzero_ps();
                __m256 sumB = _mm256_setzero_ps();


                for (int k = 0; k < n; k++) {
                    for (int l = 0; l < n; l++) {
                        if (i + k - kernel_half_size >= 0 && i + k - kernel_half_size < rows && j + l - kernel_half_size >= 0 && j + l - kernel_half_size < cols) {

                            __m256 kernel_cnst = _mm256_set1_ps(kernel[k][l]);
                            sumR = _mm256_fmadd_ps(vR, kernel_cnst, sumR);
                            sumG = _mm256_fmadd_ps(vG, kernel_cnst, sumG);
                            sumB = _mm256_fmadd_ps(vB, kernel_cnst, sumB);
                        }
                    }
                }

                __m256i red_int = _mm256_cvtps_epi32(sumR);
                __m256i green_int = _mm256_cvtps_epi32(sumG);
                __m256i blue_int = _mm256_cvtps_epi32(sumB);

                for (int k = 0; k < 8; k++)
                {
                    new_result.at<cv::Vec3b>(i, j + k) = cv::Vec3b(blue_int.m256i_i32[k], green_int.m256i_i32[k], red_int.m256i_i32[k]);
                }
            }
        }


        for (; i < this->rows; i++) {
            for (j = 0; j < this->cols; j++) {
                cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
                int blue = pixel[0];
                int green = pixel[1];
                int red = pixel[2];

                // Perform arithmetic operations on the pixel values
                int newBlue = blue - constant;
                int newGreen = green - constant;
                int newRed = red - constant;

                if (newBlue < 0) newBlue = 0;
                if (newGreen < 0) newGreen = 0;
                if (newRed < 0) newRed = 0;

                // Update the pixel values
                new_result.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(newBlue), uchar(newGreen), uchar(newRed));
            }
        }


        this->result = new_result;
        this->cols = new_result.cols;
        this->rows = new_result.rows;

        return *this;
    }

/*ResultImage& apply_optimized_filter(std::vector<std::vector<float>> kernel, int n) {
    int constant = 3;
    int stride = this->cols % 8; // if img not a multiple of 8
    Mat new_result(rows, cols, CV_8UC3);

    int kernel_half_size = n / 2;
    // __m256i kernel_half_size = _mm256_set1_epi8(static_cast<char>(n / 2));
    int i, j;
    int res = 0;


    uint8_t R[8], G[8], B[8], A[8];

    for (i = 0; i < this->rows; i++) {
        for (j = 0; j < this->cols - stride; j += 8) {
            int y = 0;

            int red_sum = 0;
            int green_sum = 0;
            int blue_sum = 0;
            // iterating through kernel and getting pixels
            for (int k = 0; k < n; k++) {
                //if (y == 8) break;

                for (int l = 0; l < n; l++) {

                    register int x = 0;
                    if (i + k - kernel_half_size >= 0 && i + k - kernel_half_size < rows && j + l - kernel_half_size >= 0 && j + l - kernel_half_size < cols) {
                        //cout <<"k " << k << " l " << l << " curr " << (k * n + l) << endl;
                        Vec3b pixels = result.at<Vec3b>(i + k - kernel_half_size, j + l - kernel_half_size);
                        //if ((k * n + l) % 8 != 0) {
                        if ((k * n + l) % 8 == 0 && (k * n + l) != 0) {
                            __m256 vR = _mm256_setr_ps(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7]);

                            __m256 vG = _mm256_setr_ps(G[0], G[1], G[2], G[3], G[4], G[5], G[6], G[7]);

                            __m256 vB = _mm256_setr_ps(B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7]);


                            __m256 sumR = _mm256_setzero_ps();
                            __m256 sumG = _mm256_setzero_ps();
                            __m256 sumB = _mm256_setzero_ps();

                            __m256 kernel_cnst = _mm256_set1_ps(kernel[k][l]);
                            sumR = _mm256_fmadd_ps(vR, kernel_cnst, sumR);
                            sumG = _mm256_fmadd_ps(vG, kernel_cnst, sumG);
                            sumB = _mm256_fmadd_ps(vB, kernel_cnst, sumB);

                            __m256i red_int = _mm256_cvtps_epi32(sumR);
                            __m256i green_int = _mm256_cvtps_epi32(sumG);
                            __m256i blue_int = _mm256_cvtps_epi32(sumB);

                            for (int x = 0; x < 8; x++){
                                red_sum += red_int.m256i_i32[x];
                                green_sum += green_int.m256i_i32[x];
                                blue_sum += blue_int.m256i_i32[x];
                            }

                            if (k == n - 1 && l == n - 1) {
                                red_sum += (float)pixels[2] * kernel[n - 1][n - 1];
                                green_sum += (float)pixels[1] * kernel[n - 1][n - 1];
                                blue_sum += (float)pixels[0] * kernel[n - 1][n - 1];
                                continue;
                            }
                            
                        }
                        cout << "ind " << (k * n + l) << endl;
                        // prvo ce biti 8 i onda ce se upisati i onda ce biti 0 i opet upisati na isto mesto
                        if((k * n + l) % 8 == 0){
                            B[(k * n + l) % 8] = pixels[x++];
                            G[(k * n + l) % 8] = pixels[x++];
                            R[(k * n + l) % 8] = pixels[x++];
                        }
                        
                        
                        //}
                        
                        //if((k * n + l) % 8)
                        //cout << "B " << B[k * n + l] << " G " << G[k * n + l] << " R " << R[k * n + l] << endl;
                    }

                }
            }*/


            /*__m256 vR = _mm256_setr_ps(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7]);

            __m256 vG = _mm256_setr_ps(G[0], G[1], G[2], G[3], G[4], G[5], G[6], G[7]);

            __m256 vB = _mm256_setr_ps(B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7]);


            __m256 sumR = _mm256_setzero_ps();
            __m256 sumG = _mm256_setzero_ps();
            __m256 sumB = _mm256_setzero_ps();*/


            /*for (int k = 0; k < n; k++) {
                for (int l = 0; l < n; l++) {
                    if (i + k - kernel_half_size >= 0 && i + k - kernel_half_size < rows && j + l - kernel_half_size >= 0 && j + l - kernel_half_size < cols) {

                        __m256 kernel_cnst = _mm256_set1_ps(kernel[k][l]);
                        sumR = _mm256_fmadd_ps(vR, kernel_cnst, sumR);
                        sumG = _mm256_fmadd_ps(vG, kernel_cnst, sumG);
                        sumB = _mm256_fmadd_ps(vB, kernel_cnst, sumB);
                    }
                }
            }*/

            /*__m256i red_int = _mm256_cvtps_epi32(sumR);
            __m256i green_int = _mm256_cvtps_epi32(sumG);
            __m256i blue_int = _mm256_cvtps_epi32(sumB);*/

            //for (int k = 0; k < 8; k++)
            //{
            //new_result.at<cv::Vec3b>(i, j) = cv::Vec3b(blue_sum, green_sum, red_sum);
            //}
        /*}
    }


    for (; i < this->rows; i++) {
        for (j = 0; j < this->cols; j++) {
            cv::Vec3b pixel = result.at<cv::Vec3b>(i, j);
            int blue = pixel[0];
            int green = pixel[1];
            int red = pixel[2];

            // Perform arithmetic operations on the pixel values
            int newBlue = blue - constant;
            int newGreen = green - constant;
            int newRed = red - constant;

            if (newBlue < 0) newBlue = 0;
            if (newGreen < 0) newGreen = 0;
            if (newRed < 0) newRed = 0;

            // Update the pixel values
            new_result.at<cv::Vec3b>(i, j) = cv::Vec3b(uchar(newBlue), uchar(newGreen), uchar(newRed));
        }
    }


    this->result = new_result;
    this->cols = new_result.cols;
    this->rows = new_result.rows;

    return *this;
}*/
};

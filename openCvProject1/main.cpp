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
#include "ResultImage.h"

using namespace cv;
using namespace std;

int get_cache_size() {
    int regs[4] = { 0 };
    __cpuid(regs, 0x80000006);
    int cache_line_size = (regs[2] & 0xFF);

    return cache_line_size;
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

vector<vector<float>> read_kernel(vector<vector<float>> kernel, int n) {

    cout << "Enter the elements of the kernel:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << "Enter element at position (" << i << ", " << j << "): ";
            std::cin >> kernel[i][j];
        }
    }

    return kernel;
}

int main() {

    // reading the image 
    //Mat img = imread("C:/Users/anjci/OneDrive/Documents/home_page_monet.jpg");
    Mat img = imread("C:/Users/anjci/OneDrive/Documents/aor22.jpg");
    Mat img2 = imread("C:/Users/anjci/OneDrive/Documents/aor22.jpg");
    //Mat img = imread("C:/Users/anjci/OneDrive/Documents/leaaor2.jpg");

    if (img.empty()) {
        cout << "Failed to load the image." << endl;
        return -1;
    }

    // taking the input
    
    int rows = img.rows;
    int cols = img.cols;

    Mat result(img.size(), img.type());
    Mat result_opt(img.size(), img.type());

    Pixel max_pixel = find_max(rows, cols, img);

    int num;

    ResultImage result_image(img);
    ResultImage result_image_opt(img2);
    while (true) {
        cout << "Enter your function: " << endl;
        cout << "0. Quit " << endl;
        cout << "1. Addition " << endl;
        cout << "2. Substraction " << endl;
        cout << "3. Inverted substraction " << endl;
        cout << "4. Multiplication " << endl;
        cout << "5. Division " << endl;
        cout << "6. Inverted division " << endl;
        cout << "7. Power " << endl;
        cout << "8. Log " << endl;
        cout << "9. Min " << endl;
        cout << "10. Max " << endl;
        cout << "11. Inversion " << endl;
        cout << "12. Grayscale " << endl;
        cout << "13. Apply filter " << endl;

        cin >> num;

        if (num == 0) {
            break;
        }
        else if (num == 1) {
            int cnst;
            cout << "Enter the constant: ";
            cin >> cnst;
            auto start = std::chrono::steady_clock::now();
            result_image.apply_add(cnst); 
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << "Time to apply the unoptimized function to a photo: " << diff.count() << " s\n";


            start = std::chrono::steady_clock::now();
            result_image_opt.apply_optimized_add(cnst); 
            end = std::chrono::steady_clock::now();
            diff = end - start;
            std::cout << "Time to apply the optimized function to a photo: " << diff.count() << " s\n";
        }
        else if (num == 2) {
            int cnst;
            cout << "Enter the constant: ";
            cin >> cnst;
            auto start = std::chrono::steady_clock::now();
            result_image.apply_sub(cnst);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << "Time to apply the unoptimized function to a photo: " << diff.count() << " s\n";


            start = std::chrono::steady_clock::now();
            result_image_opt.apply_optimized_sub(cnst);
            end = std::chrono::steady_clock::now();
            diff = end - start;
            std::cout << "Time to apply the optimized function to a photo: " << diff.count() << " s\n";
        }
        else if (num == 3) {
            int cnst;
            cout << "Enter the constant: ";
            cin >> cnst;
            auto start = std::chrono::steady_clock::now();
            result_image.apply_inverted_sub(cnst);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << "Time to apply the unoptimized function to a photo: " << diff.count() << " s\n";


            start = std::chrono::steady_clock::now();
            result_image_opt.apply_optimized_inverted_sub(cnst);
            end = std::chrono::steady_clock::now();
            diff = end - start;
            std::cout << "Time to apply the optimized function to a photo: " << diff.count() << " s\n";
        }
        else if (num == 4) {
            int cnst;
            cout << "Enter the constant: ";
            cin >> cnst;
            auto start = std::chrono::steady_clock::now();
            result_image.apply_mul(cnst);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << "Time to apply the unoptimized function to a photo: " << diff.count() << " s\n";


            start = std::chrono::steady_clock::now();
            result_image_opt.apply_optimized_mul(cnst);
            end = std::chrono::steady_clock::now();
            diff = end - start;
            std::cout << "Time to apply the optimized function to a photo: " << diff.count() << " s\n";
        }
        else if (num == 5) {
            int cnst;
            cout << "Enter the constant: ";
            cin >> cnst;
            auto start = std::chrono::steady_clock::now();
            result_image.apply_div(cnst);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << "Time to apply the unoptimized function to a photo: " << diff.count() << " s\n";


            start = std::chrono::steady_clock::now();
            result_image_opt.apply_optimized_div(cnst);
            end = std::chrono::steady_clock::now();
            diff = end - start;
            std::cout << "Time to apply the optimized function to a photo: " << diff.count() << " s\n";
        }
        else if (num == 6) {
            int cnst;
            cout << "Enter the constant: ";
            cin >> cnst;
            auto start = std::chrono::steady_clock::now();
            result_image.apply_inverted_div(cnst);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << "Time to apply the unoptimized function to a photo: " << diff.count() << " s\n";


            start = std::chrono::steady_clock::now();
            result_image_opt.apply_optimized_inverted_div(cnst);
            end = std::chrono::steady_clock::now();
            diff = end - start;
            std::cout << "Time to apply the optimized function to a photo: " << diff.count() << " s\n";
        }
        else if (num == 7) {
            int cnst;
            cout << "Enter the constant: ";
            cin >> cnst;
            auto start = std::chrono::steady_clock::now();
            result_image.apply_pow(cnst);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << "Time to apply the unoptimized function to a photo: " << diff.count() << " s\n";


            start = std::chrono::steady_clock::now();
            result_image_opt.apply_optimized_pow(cnst);
            end = std::chrono::steady_clock::now();
            diff = end - start;
            std::cout << "Time to apply the optimized function to a photo: " << diff.count() << " s\n";
        }
        else if (num == 8) {
            auto start = std::chrono::steady_clock::now();
            result_image.apply_log();
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << "Time to apply the unoptimized function to a photo: " << diff.count() << " s\n";


            start = std::chrono::steady_clock::now();
            result_image_opt.apply_optimized_log();
            end = std::chrono::steady_clock::now();
            diff = end - start;
            std::cout << "Time to apply the optimized function to a photo: " << diff.count() << " s\n";
        }
        else if (num == 9) {
            int cnst;
            cout << "Enter the constant: ";
            cin >> cnst;
            auto start = std::chrono::steady_clock::now();
            result_image.apply_min(cnst);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << "Time to apply the unoptimized function to a photo: " << diff.count() << " s\n";


            start = std::chrono::steady_clock::now();
            result_image_opt.apply_optimized_min(cnst);
            end = std::chrono::steady_clock::now();
            diff = end - start;
            std::cout << "Time to apply the optimized function to a photo: " << diff.count() << " s\n";
        }
        else if (num == 10) {
            int cnst;
            cout << "Enter the constant: ";
            cin >> cnst;
            auto start = std::chrono::steady_clock::now();
            result_image.apply_max(cnst);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << "Time to apply the unoptimized function to a photo: " << diff.count() << " s\n";


            start = std::chrono::steady_clock::now();
            result_image_opt.apply_optimized_max(cnst);
            end = std::chrono::steady_clock::now();
            diff = end - start;
            std::cout << "Time to apply the optimized function to a photo: " << diff.count() << " s\n";
        }
        else if (num == 11) {
            auto start = std::chrono::steady_clock::now();
            result_image.apply_inversion();
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << "Time to apply the unoptimized function to a photo: " << diff.count() << " s\n";


            start = std::chrono::steady_clock::now();
            result_image_opt.apply_optimized_inversion();
            end = std::chrono::steady_clock::now();
            diff = end - start;
            std::cout << "Time to apply the optimized function to a photo: " << diff.count() << " s\n";
        }
        else if (num == 12) {
            auto start = std::chrono::steady_clock::now();
            result_image.apply_grayscale();
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << "Time to apply the unoptimized function to a photo: " << diff.count() << " s\n";


            start = std::chrono::steady_clock::now();
            result_image_opt.apply_optimized_grayscale ();
            end = std::chrono::steady_clock::now();
            diff = end - start;
            std::cout << "Time to apply the optimized function to a photo: " << diff.count() << " s\n";
        }
        else if (num == 13) {
            int n;
            cout << "Insert kernel filter size: ";
            cin >> n;

            vector<vector<float>> kernel(n, vector<float>(n));
            kernel = read_kernel(kernel, n);

            auto start = std::chrono::steady_clock::now();

            result_image.apply_filter(kernel, n);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << "Time to apply the unoptimized function to a photo: " << diff.count() << " s\n";


            start = std::chrono::steady_clock::now();
            result_image_opt.apply_optimized_filter(kernel, n);
            end = std::chrono::steady_clock::now();
            diff = end - start;
            std::cout << "Time to apply the optimized function to a photo: " << diff.count() << " s\n";
        }

        result = result_image.get_result();
        result_opt = result_image_opt.get_result();

        imshow("Result", result);
        cv::waitKey(0);
        cv::destroyAllWindows();


        imshow("Result opt", result_opt);
        cv::waitKey(0);
        cv::destroyAllWindows();

    }
    


    auto start = std::chrono::steady_clock::now();
    //result_image.apply_add(50); result = result_image.get_result();
    //result_image.apply_sub(50); result = result_image.get_result();
    //result_image.apply_inverted_sub(150); result = result_image.get_result();
    //result_image.apply_mul(5); result = result_image.get_result();
    //result_image.apply_div(4); result = result_image.get_result();
    //result_image.apply_inverted_div(255); result = result_image.get_result();
    //result_image.apply_pow(3); result = result_image.get_result();
    //result_image.apply_log(); result = result_image.get_result();
    //result_image.apply_min(135); result = result_image.get_result();
    //result_image.apply_max(115); result = result_image.get_result();
    //result_image.apply_inversion(); result = result_image.get_result();
    result_image.apply_grayscale(); result = result_image.get_result();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time to apply the unoptimized function to a photo: " << diff.count() << " s\n";

    imshow("Result", result);
    cv::waitKey(0);
    cv::destroyAllWindows();

    start = std::chrono::steady_clock::now();
    //result_image_opt.apply_optimized_add(50); result_opt = result_image_opt.get_result();
    //result_image_opt.apply_optimized_sub(50); result_opt = result_image_opt.get_result();
    //result_image_opt.apply_optimized_inverted_sub(150); result_opt = result_image_opt.get_result();
    //result_image_opt.apply_optimized_mul(5); result_opt = result_image_opt.get_result();
    //result_image_opt.apply_optimized_div(4); result_opt = result_image_opt.get_result();
    //result_image_opt.apply_optimized_inverted_div(255); result_opt = result_image_opt.get_result();
    //result_image_opt.apply_optimized_pow(3); result_opt = result_image_opt.get_result();
    //result_image_opt.apply_optimized_log(); result_opt = result_image_opt.get_result();

    //result_image_opt.apply_optimized_min(135); result_opt = result_image_opt.get_result();

    //result_image_opt.apply_optimized_max(115); result_opt = result_image_opt.get_result();

    //result_image_opt.apply_optimized_inversion(); result_opt = result_image_opt.get_result();

    result_image_opt.apply_optimized_grayscale(); result_opt = result_image_opt.get_result();

    end = std::chrono::steady_clock::now();
    diff = end - start;
    std::cout << "Time to apply the unoptimized function to a photo: " << diff.count() << " s\n";

    imshow("Result opt", result_opt);
    cv::waitKey(0);
    cv::destroyAllWindows();

	return 0;
}
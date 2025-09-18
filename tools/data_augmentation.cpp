#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>
#include <vector>
#include <string>
#include <random>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// rotacionar imagem
Mat rotateImage(const Mat& src, double angle) {
    Point2f center(src.cols/2.0F, src.rows/2.0F);
    Mat rot = getRotationMatrix2D(center, angle, 1.0);
    Mat dst;
    warpAffine(src, dst, rot, src.size());
    return dst;
}

// Espelhamento
Mat flipImage(const Mat& src, int flipCode) {
    Mat dst;
    flip(src, dst, flipCode);
    return dst;
}

// Zoom (crop central e resize)
Mat zoomImage(const Mat& src, double zoomFactor) {
    int h = src.rows, w = src.cols;
    int nh = int(h / zoomFactor), nw = int(w / zoomFactor);
    int y1 = (h - nh) / 2, x1 = (w - nw) / 2;
    Rect roi(x1, y1, nw, nh);
    Mat cropped = src(roi);
    Mat dst;
    resize(cropped, dst, src.size());
    return dst;
}

// RGB Shift
Mat rgbShift(const Mat& src, int r_shift, int g_shift, int b_shift) {
    vector<Mat> channels;
    split(src, channels);
    channels[2] = channels[2] + r_shift; // R
    channels[1] = channels[1] + g_shift; // G
    channels[0] = channels[0] + b_shift; // B
    Mat dst;
    merge(channels, dst);
    return dst;
}

// HueSaturationValue
Mat hueSaturationValue(const Mat& src, int hue_shift, double sat_mult, double val_mult) {
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);
    vector<Mat> channels;
    split(hsv, channels);
    channels[0] = channels[0] + hue_shift;
    for (int y = 0; y < channels[0].rows; ++y) {
        for (int x = 0; x < channels[0].cols; ++x) {
            int val = channels[0].at<uchar>(y, x);
            val = (val + 180) % 180;
            channels[0].at<uchar>(y, x) = static_cast<uchar>(val);
        }
    }
    channels[1] = channels[1] * sat_mult;
    channels[2] = channels[2] * val_mult;
    merge(channels, hsv);
    Mat dst;
    cvtColor(hsv, dst, COLOR_HSV2BGR);
    return dst;
}

// Channel Shuffle
Mat channelShuffle(const Mat& src) {
    vector<Mat> channels;
    split(src, channels);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(channels.begin(), channels.end(), g);
    Mat dst;
    merge(channels, dst);
    return dst;
}

// CLAHE
Mat applyCLAHE(const Mat& src) {
    Mat lab;
    cvtColor(src, lab, COLOR_BGR2Lab);
    vector<Mat> lab_channels;
    split(lab, lab_channels);
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4.0);
    clahe->apply(lab_channels[0], lab_channels[0]);
    merge(lab_channels, lab);
    Mat dst;
    cvtColor(lab, dst, COLOR_Lab2BGR);
    return dst;
}

// Random Contrast
Mat randomContrast(const Mat& src, double alpha) {
    Mat dst;
    src.convertTo(dst, -1, alpha, 0);
    return dst;
}

// Random Gamma
Mat randomGamma(const Mat& src, double gamma) {
    Mat lut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; i++)
        lut.at<uchar>(i) = pow(i / 255.0, gamma) * 255.0;
    Mat dst;
    LUT(src, lut, dst);
    return dst;
}

// Random Brightness
Mat randomBrightness(const Mat& src, int beta) {
    Mat dst;
    src.convertTo(dst, -1, 1, beta);
    return dst;
}

// Blur
Mat applyBlur(const Mat& src, int ksize) {
    Mat dst;
    blur(src, dst, Size(ksize, ksize));
    return dst;
}

// Median Blur
Mat applyMedianBlur(const Mat& src, int ksize) {
    Mat dst;
    medianBlur(src, dst, ksize);
    return dst;
}

// ToGray
Mat toGray(const Mat& src) {
    Mat dst;
    cvtColor(src, dst, COLOR_BGR2GRAY);
    return dst;
}

// JPEG Compression (simulado: salvar e recarregar)
Mat jpegCompression(const Mat& src, int quality) {
    vector<uchar> buf;
    vector<int> params = {IMWRITE_JPEG_QUALITY, quality};
    imencode(".jpg", src, buf, params);
    return imdecode(buf, IMREAD_COLOR);
}

// Split canais RGB
vector<Mat> splitChannels(const Mat& src) {
    vector<Mat> channels;
    split(src, channels);
    return channels;
}

void saveImage(const Mat& img, const string& path) {
    if (imwrite(path, img)) {
        printf("Salvo: %s\n", path.c_str());
    } else {
        printf("ERRO ao salvar: %s\n", path.c_str());
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Uso: %s <diretorio_entrada> <diretorio_saida>\n", argv[0]);
        return 1;
    }
    string input_dir = argv[1];
    string output_dir = argv[2];
    fs::create_directories(output_dir);

    vector<double> angles = {30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360};
    vector<double> zooms = {1.2, 1.5, 2.0};

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            Mat img = imread(entry.path().string());
            if (img.empty()) continue;
            string base = entry.path().stem().string();

            // Rotations
            for (double angle : angles) {
                Mat rot = rotateImage(img, angle);
                saveImage(rot, output_dir + "/" + base + "_rot_" + to_string(int(angle)) + ".jpg");
            }

            // Zooms
            for (double z : zooms) {
                Mat zoomed = zoomImage(img, z);
                saveImage(zoomed, output_dir + "/" + base + "_zoom_" + to_string(int(z*10)) + ".jpg");
            }

            // Flips
            saveImage(flipImage(img, 0), output_dir + "/" + base + "_flip_v.jpg");
            saveImage(flipImage(img, 1), output_dir + "/" + base + "_flip_h.jpg");
            saveImage(flipImage(img, -1), output_dir + "/" + base + "_flip_both.jpg");

            // RGB Shift
            saveImage(rgbShift(img, 30, 0, 0), output_dir + "/" + base + "_rgbshift_r.jpg");
            saveImage(rgbShift(img, 0, 30, 0), output_dir + "/" + base + "_rgbshift_g.jpg");
            saveImage(rgbShift(img, 0, 0, 30), output_dir + "/" + base + "_rgbshift_b.jpg");

            // HueSaturationValue
            saveImage(hueSaturationValue(img, 15, 1.2, 1.2), output_dir + "/" + base + "_hsv.jpg");

            // Channel Shuffle
            saveImage(channelShuffle(img), output_dir + "/" + base + "_chshuffle.jpg");

            // CLAHE
            saveImage(applyCLAHE(img), output_dir + "/" + base + "_clahe.jpg");

            // Random Contrast
            saveImage(randomContrast(img, 1.5), output_dir + "/" + base + "_contrast.jpg");

            // Random Gamma
            saveImage(randomGamma(img, 0.5), output_dir + "/" + base + "_gamma.jpg");

            // Random Brightness
            saveImage(randomBrightness(img, 50), output_dir + "/" + base + "_bright.jpg");

            // Blur
            saveImage(applyBlur(img, 5), output_dir + "/" + base + "_blur.jpg");

            // Median Blur
            saveImage(applyMedianBlur(img, 5), output_dir + "/" + base + "_medblur.jpg");

            // ToGray
            saveImage(toGray(img), output_dir + "/" + base + "_gray.jpg");

            // JPEG Compression
            saveImage(jpegCompression(img, 50), output_dir + "/" + base + "_jpeg50.jpg");

            // Split canais RGB
            auto chans = splitChannels(img);
            saveImage(chans[0], output_dir + "/" + base + "_R.jpg");
            saveImage(chans[1], output_dir + "/" + base + "_G.jpg");
            saveImage(chans[2], output_dir + "/" + base + "_B.jpg");

            // Translações
            int t = 50;
            Mat M_left = (Mat_<float>(2, 3) << 1, 0, -t, 0, 1, 0);
            Mat M_right = (Mat_<float>(2, 3) << 1, 0, t, 0, 1, 0);
            Mat M_up = (Mat_<float>(2, 3) << 1, 0, 0, 0, 1, -t);
            Mat M_down = (Mat_<float>(2, 3) << 1, 0, 0, 0, 1, t);
            Mat img_left, img_right, img_up, img_down;
            warpAffine(img, img_left, M_left, img.size());
            warpAffine(img, img_right, M_right, img.size());
            warpAffine(img, img_up, M_up, img.size());
            warpAffine(img, img_down, M_down, img.size());
            saveImage(img_left, output_dir + "/" + base + "_trans_left.jpg");
            saveImage(img_right, output_dir + "/" + base + "_trans_right.jpg");
            saveImage(img_up, output_dir + "/" + base + "_trans_up.jpg");
            saveImage(img_down, output_dir + "/" + base + "_trans_down.jpg");
        }
    }
    printf("Aumentação concluída\n");
    return 0;
}
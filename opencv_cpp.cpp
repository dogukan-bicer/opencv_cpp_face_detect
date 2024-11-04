#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

using namespace cv;
using namespace std;
using namespace dnn;

const double CONFIDENCE_THRESHOLD = 0.5;
const double NMS_THRESHOLD = 0.1;
const int FRAME_THICKNESS = 2;  // Çerçeve kalınlığı ayarı
const int FONT_THICKNESS = 2;   // Yazı kalınlığı
const int MAX_WIDTH = 600;

// Darknet model dosyalarının yolları
const string model = "yolov4-tiny-obj_final.weights";
const string config = "yolov4-tiny-obj.cfg";
const string coco_names = "coco.names";

vector<string> loadClassNames(const string& file_path) {
    vector<string> class_names;
    ifstream ifs(file_path.c_str());
    string line;
    while (getline(ifs, line)) class_names.push_back(line);
    return class_names;
}

int main() {
    vector<string> class_names = loadClassNames(coco_names);
    Net net = readNetFromDarknet(config, model);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    VideoCapture video_capture(0);
    if (!video_capture.isOpened()) {
        cerr << " webcam acilmiyor!" << endl;
        return -1;
    }

    // Webcam çözünürlüğünü artırma
    video_capture.set(CAP_PROP_FRAME_WIDTH, 1280);
    video_capture.set(CAP_PROP_FRAME_HEIGHT, 720);

    cout << "Gercek zamanlı goruntu tespiti Basliyor..." << endl;
    auto start_time = chrono::steady_clock::now();
    int frame_count = 0;
    double fps = 0.0;

    while (true) {
        Mat frame;
        video_capture >> frame;
        if (frame.empty()) break;

        Mat blob = blobFromImage(frame, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        vector<Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        vector<int> class_ids;
        vector<float> confidences;
        vector<Rect> boxes;

        // Çıkış katmanı analizi
        for (auto& output : outputs) {
            for (int i = 0; i < output.rows; i++) {
                float confidence = output.at<float>(i, 4);
                if (confidence > CONFIDENCE_THRESHOLD) {
                    Mat scores = output.row(i).colRange(5, output.cols);
                    Point class_id_point;
                    double max_class_score;
                    minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);
                    if (max_class_score > CONFIDENCE_THRESHOLD) {
                        int width = static_cast<int>(output.at<float>(i, 2) * frame.cols);
                        if (width < MAX_WIDTH)
                        {
                            int center_x = static_cast<int>(output.at<float>(i, 0) * frame.cols);
                            int center_y = static_cast<int>(output.at<float>(i, 1) * frame.rows);
                            int height = static_cast<int>(output.at<float>(i, 3) * frame.rows);
                            int left = center_x - width / 2;
                            int top = center_y - height / 2;

                            class_ids.push_back(class_id_point.x);
                            confidences.push_back(static_cast<float>(max_class_score));
                            boxes.push_back(Rect(left, top, width, height));
                        }

                    }
                }
            }
        }

        // NMS ile gereksiz kareleri ortadan kaldırma
        vector<int> indices;
        NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

        

        // Eğer geçerli bir nesne algılandıysa kare çiz
        if (!indices.empty()) {
            for (int idx : indices) {
                Rect box = boxes[idx];
                rectangle(frame, box, Scalar(0, 255, 0), FRAME_THICKNESS);

                string label = class_names[class_ids[idx]] + " (" + to_string(static_cast<int>(confidences[idx] * 100)) + "%)";
                int baseline;
                Size label_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, FONT_THICKNESS, &baseline);
                rectangle(frame, Point(box.x, box.y - label_size.height - 10), Point(box.x + label_size.width, box.y), Scalar(0, 255, 0), FILLED);
                putText(frame, label, Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), FONT_THICKNESS);
            }
        }

        frame_count++;
        auto end_time = chrono::steady_clock::now();
        double elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
        if (elapsed_time > 1000) {
            fps = frame_count / (elapsed_time / 1000.0);
            frame_count = 0;
            start_time = end_time;
        }

        // FPS bilgisini ekrana yazdır
        putText(frame, "FPS: " + to_string(static_cast<int>(fps)), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), FONT_THICKNESS);

        imshow("Nesne tespiti", frame);
        if (waitKey(1) == 27) break;  // Çıkmak için 'Esc' tuşuna basın
    }

    video_capture.release();
    destroyAllWindows();
    return 0;
}

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp> // DNN modülü için
#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>


using namespace cv;
using namespace std;
using namespace dnn; // DNN isim alanını kullan

const double TOLERANCE = 25000;    // Yüz karşılaştırma toleransı
const int FRAME_THICKNESS = 3;
const int FONT_THICKNESS = 2;

// DNN model dosyalarının yolu
const string model = "res10_300x300_ssd_iter_140000.caffemodel";
const string config = "deploy.prototxt";

vector<string> GetFilesInDirectory(const string& dir) {
    vector<string> files;
    WIN32_FIND_DATAA file_data;  // 'A' versiyonu, dar karakterli
    HANDLE hFind = FindFirstFileA((dir + "\\*").c_str(), &file_data);  // 'A' versiyonu, dar karakterli

    if (hFind == INVALID_HANDLE_VALUE) {
        cerr << "Klasör bulunamadı: " << dir << endl;
        return files;
    }

    do {
        const string file_name = file_data.cFileName;  // Dosya adı
        const string full_file_name = dir + "\\" + file_name;
        const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

        if (file_name[0] == '.') {
            continue;  // Gizli dosyaları atla
        }

        if (!is_directory) {
            files.push_back(full_file_name);  // Dosya adını listeye ekle
        }
    } while (FindNextFileA(hFind, &file_data) != 0);  // 'A' versiyonu, dar karakterli

    FindClose(hFind);
    return files;
}

int main() {
    // DNN modeli yükle
    Net net = readNetFromCaffe(config, model);
    if (net.empty()) {
        cerr << "Yuz algılayıcı yüklenemedi!" << endl;
        return -1;
    }

    cout << "Opencv kutuphanesi yuklendi!!!\n";
    cout << CV_VERSION << std::endl;

    // Yüz veritabanı için örnek resimler ve etiketler
    vector<Mat> known_faces;
    vector<string> known_names;

    // Dostların yüz verilerini yükle
    string known_faces_dir = "dostlar";
    vector<string> files = GetFilesInDirectory(known_faces_dir);

    for (const string& file_path : files) {
        Mat img = imread(file_path, IMREAD_COLOR);  // Renkli olarak yükle
        if (!img.empty()) {
            // Yüz ROI'yi algıla ve veritabanına ekle
            Mat blob = blobFromImage(img, 1.0, Size(300, 300), Scalar(104, 117, 123), false, false);
            net.setInput(blob);
            Mat detections = net.forward();

            // Detections matrisinden verileri al
            const float* data = detections.ptr<float>();
            for (int i = 0; i < detections.size[2]; i++) {
                float confidence = data[i * 7 + 2]; // Güven eşiği
                if (confidence > 0.5) { // Güven eşiği
                    int x1 = static_cast<int>(data[i * 7 + 3] * img.cols);
                    int y1 = static_cast<int>(data[i * 7 + 4] * img.rows);
                    int x2 = static_cast<int>(data[i * 7 + 5] * img.cols);
                    int y2 = static_cast<int>(data[i * 7 + 6] * img.rows);

                    Mat faceROI = img(Rect(Point(x1, y1), Point(x2, y2)));  // Yüz ROI
                    known_faces.push_back(faceROI);
                    string file_name = file_path.substr(file_path.find_last_of("\\") + 1);
                    known_names.push_back(file_name.substr(0, file_name.find_last_of(".")));
                    cout << "Tespit edilecek yuz: " << known_names.back() << std::endl;
                }
            }
        }
    }

    // Web kamerasını başlat
    VideoCapture video_capture(0);
    if (!video_capture.isOpened()) {
        cerr << "Web kamerası açılamadı!" << endl;
        return -1;
    }

    cout << "Gercek zamanli yuz tanima basliyor..." << endl;

    // FPS hesaplamak için zamanlayıcı
    auto start_time = chrono::steady_clock::now();
    int frame_count = 0;

    while (true) {
        Mat frame;
        video_capture >> frame;

        if (frame.empty()) break;

        // Yüzleri algıla
        Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 117, 123), false, false);
        net.setInput(blob);
        Mat detections = net.forward();

        // Detections matrisinden verileri al
        const float* data = detections.ptr<float>();
        for (int i = 0; i < detections.size[2]; i++) {
            float confidence = data[i * 7 + 2];
            if (confidence > 0.5) { // Güven eşiği
                int x1 = static_cast<int>(data[i * 7 + 3] * frame.cols);
                int y1 = static_cast<int>(data[i * 7 + 4] * frame.rows);
                int x2 = static_cast<int>(data[i * 7 + 5] * frame.cols);
                int y2 = static_cast<int>(data[i * 7 + 6] * frame.rows);

                Mat faceROI = frame(Rect(Point(x1, y1), Point(x2, y2))); // ROI'yi al

                // Yüzü karşılaştır
                string name = "Bilinmeyen";
                double best_confidence = 1.0;  // En düşük benzerlik skoru
                double confidence_percentage = 0.0;

                for (size_t i = 0; i < known_faces.size(); i++) {
                    Mat known_face_resized;
                    resize(known_faces[i], known_face_resized, faceROI.size());
                    double confidence = norm(known_face_resized, faceROI, NORM_L2);
                    confidence_percentage = -100.0 * (1 - confidence / TOLERANCE); // Yüzde olarak hesapla
                    if (confidence > TOLERANCE) {
                        name = known_names[i];
                        break; // Eşleşme bulunduğunda döngüyü kır
                    }
                }

                // Çerçeve rengini belirle
                Scalar color = (name == "Bilinmeyen") ? Scalar(0, 0, 255) : Scalar(0, 255, 0); // Kırmızı veya yeşil

                // Yüzü kare içine alın ve isim yazın
                rectangle(frame, Rect(Point(x1, y1), Point(x2, y2)), color, FRAME_THICKNESS);
                putText(frame, name, Point(x1, y1 - 10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), FONT_THICKNESS);
                putText(frame, to_string(confidence_percentage) + "%", Point(x1, y2 + 10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), FONT_THICKNESS);
            }
        }

        // FPS hesapla
        frame_count++;
        auto end_time = chrono::steady_clock::now();
        double elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
        if (elapsed_time > 1000) { // 1 saniyede bir FPS güncelle
            double fps = frame_count / (elapsed_time / 1000.0);
            frame_count = 0;
            start_time = end_time;

            // FPS değerini ekranda göster
            putText(frame, "FPS: " + to_string(static_cast<int>(fps)), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), FONT_THICKNESS);
        }

        // FPS değerini sürekli güncelle
        putText(frame, "FPS: " + to_string(static_cast<int>(frame_count * 1000.0 / elapsed_time)), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), FONT_THICKNESS);

        // Görüntüyü göster
        imshow("Yuz Tanima", frame);

        // Çıkış tuşu
        if (waitKey(1) == 27) break; // ESC tuşu ile çık
    }

    video_capture.release();
    destroyAllWindows();
    return 0;
}

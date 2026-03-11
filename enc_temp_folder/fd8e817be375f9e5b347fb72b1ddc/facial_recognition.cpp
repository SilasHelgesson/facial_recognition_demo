#include <iostream>
#include <vector>
#include <string> 
#include <opencv2/opencv.hpp>

constexpr int NUM_IMAGES_PER_PERSON = 10;
constexpr int NUM_PEOPLE = 15;
constexpr int DIMENSION_X = 100;
constexpr int DIMENSION_Y = 100;
constexpr bool DISPLAY_FACES = true;
constexpr const char* PATH_TO_TRAINING_DATA = "C:/Users/DrBre/source/repos/facial_recognition/facial_recognition/data/train/";
constexpr const char* PATH_TO_TESTING_DATA = "C:/Users/DrBre/source/repos/facial_recognition/facial_recognition/data/test/";

struct facial_data
{
    std::vector<cv::Mat> image_data;
    std::vector<int> labels;
    std::vector<std::string> names;
    cv::Mat mean_face;
    cv::Mat training_data;
    cv::Mat centered_data;
    cv::Mat eigenfaces;
};

struct prediction_result
{
    int predicted_label;
    double confidence;
};
std::string convert_number_to_double_digit_string(int number) {
    if (number < 10) {
        return "0" + std::to_string(number);
    }
    return std::to_string(number);
}

void re_size_images(std::vector<cv::Mat>& images) {
    for (auto& image : images) {
        cv::resize(image, image, cv::Size(DIMENSION_X, DIMENSION_Y));
    }
}

cv::Mat create_data_matrix(const std::vector<cv::Mat>& images) {
    int num_images = images.size();
    int image_dimensions = DIMENSION_X * DIMENSION_Y;

    cv::Mat training_data(num_images, image_dimensions, CV_32F);

    for (int i = 0; i < num_images; i++) {
        cv::Mat image_row = training_data.row(i);
        cv::Mat image_col = images[i].reshape(1, 1);
        image_col.convertTo(image_row, CV_32F);
    }
    return training_data;
}

cv::Mat get_mean_face(const std::vector<cv::Mat>& images) {
    cv::Mat mean_face = cv::Mat::zeros(1, DIMENSION_X * DIMENSION_Y, CV_32F);
    for (size_t i = 0; i < images.size(); i++) {
        cv::Mat img_row;
        images[i].reshape(1, 1).convertTo(img_row, CV_32F);
        mean_face += img_row;
    }

    mean_face /= images.size();
    return mean_face;
}

cv::Mat center_data(const cv::Mat& training_data, const cv::Mat& mean_face) {
    cv::Mat centered = training_data.clone();
    for (int i = 0; i < centered.rows; i++) {
        for (int j = 0; j < centered.cols; j++) {
            centered.at<float>(i, j) -= mean_face.at<float>(0, j);
        }
    }
    return centered;
}

void prepare_labels(facial_data& data) {
    data.names = { "Gordon Freeman","Alyx Vance","Eli Vance","Isaac Kleiner","Barney Calhoun","Judith Mossman","Wallace Breen","G-Man","Adrian Shephard","Odessa Cubbage","Arne Magnusson","Father Grigori","Russell","Laszlo","Dog" };
    data.labels.reserve(data.names.size() * NUM_IMAGES_PER_PERSON);
    for (size_t i = 0; i < data.names.size(); i++) {
        data.labels.insert(data.labels.end(), NUM_IMAGES_PER_PERSON, i);
    }
}

void read_training_images(facial_data& data) {
    for (size_t i = 0; i < data.names.size(); i++) {
        for (size_t j = 0; j < NUM_IMAGES_PER_PERSON; j++) {
            std::string image_path = PATH_TO_TRAINING_DATA + convert_number_to_double_digit_string(i + 1) + "_" + convert_number_to_double_digit_string(j) + ".png";
            cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
            if (image.empty()) {
                std::cerr << "OOPS ! Could not read image: " << image_path << std::endl;
                std::exit(1);
            }
            data.image_data.push_back(image);
        }
    }
}

void display_image(const cv::Mat& image, const std::string& window_name) {
    cv::Mat display_img;
    cv::normalize(image, display_img, 0, 255, cv::NORM_MINMAX);
    display_img.convertTo(display_img, CV_8U);
    cv::imshow(window_name, display_img);
    cv::waitKey(0);
}

void compute_eigenfaces(facial_data& data) {
    cv::Mat cov = data.centered_data * data.centered_data.t();
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(cov, eigenvalues, eigenvectors);

    data.eigenfaces = data.centered_data.t() * eigenvectors.t();
}

prediction_result predict(const facial_data& data, const cv::Mat& test_image,cv::Mat &omega_test,cv::Mat &training_omegas) {
	prediction_result prediction = { -1, DBL_MAX };

    for (size_t i = 0; i < training_omegas.rows; i++) {
        double dist = cv::norm(omega_test, training_omegas.row(i), cv::NORM_L2);
        if (dist < prediction.confidence) {
            prediction.confidence = dist;
            prediction.predicted_label = i;
        }
    }
    return prediction;
}

int main()
{
    facial_data data;

    prepare_labels(data);
    read_training_images(data);
    re_size_images(data.image_data);

    data.training_data = create_data_matrix(data.image_data);
    data.mean_face = get_mean_face(data.image_data);
    data.centered_data = center_data(data.training_data, data.mean_face);

    if (DISPLAY_FACES) {
        cv::Mat mean_img = data.mean_face.reshape(1, DIMENSION_Y);
        display_image(mean_img, "Mean Face");
    }

    compute_eigenfaces(data);

    if (DISPLAY_FACES) {
        for (size_t i = 0; i < std::min(15, data.eigenfaces.cols); i++) {
            cv::Mat ef = data.eigenfaces.col(i).clone();
            cv::Mat ef_img = ef.reshape(1, DIMENSION_Y);
            display_image(ef_img, "Eigenface " + std::to_string(i + 1));
        }
    }

    int successful_predictions = 0;
    for (const std::string& name : data.names) {
        cv::Mat test_img = cv::imread(PATH_TO_TESTING_DATA + name + ".png", cv::IMREAD_GRAYSCALE);
        if (test_img.empty()) {
            std::cerr << "OOPS ! Could not read image: " << PATH_TO_TESTING_DATA + name + ".png" << std::endl;
            std::exit(1);
        }
        cv::resize(test_img, test_img, cv::Size(DIMENSION_X, DIMENSION_Y));
        cv::Mat test_row;
        test_img.reshape(1, 1).convertTo(test_row, CV_32F);

        cv::Mat phi_test = test_row - data.mean_face;
        cv::Mat omega_test = phi_test * data.eigenfaces;
        cv::Mat training_omegas = data.centered_data * data.eigenfaces;
		prediction_result result = predict(data, test_row, omega_test, training_omegas);
        
        if (name == data.names[data.labels[result.predicted_label]]) {
            successful_predictions++;
        }
        std::cout << "Person: " << name << " Predicted person: " << data.names[data.labels[result.predicted_label]]
            << " (distance = " << result.confidence << ")" << std::endl;
    }
    std::cout << "Successful predictions: " << successful_predictions << "/" << data.names.size() << std::endl;
}
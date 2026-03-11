#include <iostream>
#include <vector>
#include <string> 
#include <opencv2/opencv.hpp>

#define NUM_IMAGES_PER_PERSON 10
#define NUM_PEOPLE 15
#define DIMENSION_X 100
#define DIMENSION_Y 100
#define DISPLAY_FACES false 

struct facial_data
{
    std::vector<cv::Mat> image_data;
    std::vector<int> labels;
    std::vector<std::string> names;
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
    cv::Mat mean_face = cv::Mat::zeros(1, DIMENSION_X*DIMENSION_Y, CV_32F);
    for (size_t i = 0; i < images.size(); i++) {
        cv::Mat img_row;
        images[i].reshape(1, 1).convertTo(img_row, CV_32F); // flatten and convert to float
        mean_face += img_row; // add whole flattened image
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

int main()
{
	facial_data data;
   
    data.names = { "Gordon Freeman","Alyx Vance","Eli Vance","Isaac Kleiner","Barney Calhoun","Judith Mossman","Wallace Breen","G-Man","Adrian Shephard","Odessa Cubbage","Arne Magnusson","Father Grigori","Russell","Laszlo","Dog" };
    data.labels.reserve(data.names.size() * NUM_IMAGES_PER_PERSON);
    for (size_t i = 0; i < data.names.size(); i++) {
        data.labels.insert(data.labels.end(), NUM_IMAGES_PER_PERSON, i);
    }

    for (size_t i = 0; i < data.names.size(); i++) {
        for(size_t j = 0; j < NUM_IMAGES_PER_PERSON; j++) {
            std::string image_path = "C:/Users/DrBre/source/repos/facial_recognition/facial_recognition/data/train/" + convert_number_to_double_digit_string(i+1) + "_" + convert_number_to_double_digit_string(j) + ".png";
            cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
            if (image.empty()) {
                std::cerr << "OOPS ! Could not read image: " << image_path << std::endl;
                continue;
            }
            data.image_data.push_back(image);
		}
    }

    re_size_images(data.image_data); // doing this in place by reference
    cv::Mat training_data = create_data_matrix(data.image_data);
	cv::Mat mean_face = get_mean_face(data.image_data);
    cv::Mat centered_data = center_data(training_data, mean_face);


    //Show your beutiful face 
    cv::Mat mean_face_img = mean_face.reshape(1, DIMENSION_Y);
    cv::normalize(mean_face_img, mean_face_img, 0, 255, cv::NORM_MINMAX);
    mean_face_img.convertTo(mean_face_img, CV_8U);
    if (DISPLAY_FACES) {
        cv::imshow("Mean Face", mean_face_img);
        cv::waitKey(0);

    }

    cv::Mat cov = centered_data * centered_data.t();
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(cov, eigenvalues, eigenvectors);

    cv::Mat eigenfaces = centered_data.t() * eigenvectors.t();

    for (int i = 0; i < eigenfaces.cols; i++) {
        cv::Mat ef = eigenfaces.col(i).clone();
        cv::normalize(ef, ef, 0, 255, cv::NORM_MINMAX);
        cv::Mat ef_img = ef.reshape(1, DIMENSION_Y);
        ef_img.convertTo(ef_img, CV_8U);
        if(DISPLAY_FACES) {
            cv::imshow("Eigenface " + std::to_string(i + 1), ef_img);
            cv::waitKey(0); 
		}
    }
	int successful_predictions = 0;
	for (std::string name : data.names) {
        cv::Mat test_img = cv::imread("C:/Users/DrBre/source/repos/facial_recognition/facial_recognition/data/test/" + name + ".png", cv::IMREAD_GRAYSCALE);
        cv::resize(test_img, test_img, cv::Size(DIMENSION_X, DIMENSION_Y));
        cv::Mat test_row;
        test_img.reshape(1, 1).convertTo(test_row, CV_32F);

        cv::Mat phi_test = test_row - mean_face;
        cv::Mat omega_test = phi_test * eigenfaces;

        int closest_idx = 0;
        double min_dist = DBL_MAX;
        for (int i = 0; i < centered_data.rows; i++) {
            cv::Mat omega_train = centered_data.row(i) * eigenfaces;
            double dist = cv::norm(omega_test, omega_train, cv::NORM_L2);
            if (dist < min_dist) {
                min_dist = dist;
                closest_idx = i;
            }
        }
		if (name == data.names[data.labels[closest_idx]]) {
            successful_predictions++;
        }
        std::cout << "Person: " << name << " Predicted person: " << data.names[data.labels[closest_idx]]
            << " (distance = " << min_dist << ")" << std::endl;
    }
	std::cout << "Successful predictions: " << successful_predictions << "/" << data.names.size() << std::endl;
}


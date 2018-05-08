#include <iostream>
#include <random>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>

#include "ldnn/network.hpp"

template<class T, class URBG>
auto create_square(size_t count, T size, URBG&& gen)
{
    using classification = ldnn::network<double>::classification;

    std::uniform_real_distribution<> dis(-size / 2, size / 2);
    auto create_example = [&]() {
        auto v = ldnn::vector<double>{dis(gen), dis(gen)};
        return classification{v,
            std::abs(v[0]) <= size / 4 && std::abs(v[1]) <= size / 4};
    };

    auto result = std::vector<classification>(count);
    util::generate(result, create_example);
    return result;
}

template<class T>
auto show_result(ldnn::network<T> network,
    cv::Rect_<T> index_space, cv::Size image_size) {

    auto clf = std::vector<T>(image_size.width * image_size.height);
    for (auto y : indices(image_size.height)) {
        for (auto x : indices(image_size.width)) {
            clf[x + y * image_size.width] = network.classify(
                ldnn::vector<double>{
                    index_space.x +
                        x / static_cast<T>(image_size.width) * index_space.width,
                    index_space.y +
                        y / static_cast<T>(image_size.height) * index_space.height});
        }
    }

    auto [min, max] = util::minmax(clf);
    auto image = cv::Mat(image_size, CV_8UC1);
    for (int y : indices(image_size.height)) {
        for (int x : indices(image_size.width)) {
            image.at<uchar>(y, x) = static_cast<uchar>(
                255.0 * (clf[x + y * image_size.width] - min) / (max - min));
        }
    }

    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display window", image);
    cv::imwrite("result.png", image);
}

int main() {
    auto start_time = std::chrono::system_clock::now();

    std::random_device rd;
    std::mt19937 gen(rd());

    std::cout << "creating data...\r" << std::flush;
    auto examples = create_square<double>(1500, 4, gen);

    std::cout << "initializing...\r" << std::flush;
    auto network = ldnn::network<double>(4, 4, 5, examples, gen);

    for (auto step [[maybe_unused]] : indices(10)) {
        util::shuffle(examples, gen);
        network.gradient_descent(examples);
    }

    std::cout << "rendering...\r" << std::flush;
    show_result(network,
        cv::Rect_<double>(-2., -2., 4., 4.),
        cv::Size(1000, 1000));

    std::cout << "done! ("
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now() - start_time).count()
              << "ms)\n";

    while (cv::waitKey(0) != 'c');
}

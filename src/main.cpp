#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <range/v3/algorithm/minmax.hpp>
#include <range/v3/view/generate.hpp>

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
    return ranges::view::generate(create_example)
        | ranges::view::take(count)
        | ranges::to_<std::vector<classification>>();
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

    auto [min, max] = ranges::minmax(clf);
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
    std::random_device rd;
    std::mt19937 gen(rd());

    auto examples = create_square<double>(1500, 4, gen);

    std::cout << "initializing...\n";
    auto network = ldnn::network<double>(4, 4, 5, examples, gen);

    for (auto step [[maybe_unused]] : indices(10)) {
        ranges::shuffle(examples, gen);
        network.gradient_descent(examples);
    }

    std::cout << "rendering...\n";
    show_result(network,
        cv::Rect_<double>(-2., -2., 4., 4.),
        cv::Size(1000, 1000));

    while (cv::waitKey(0) != 'c');
}



//特征点检测及匹配  成功实现
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    // 打开默认摄像头
    VideoCapture cap(1+CAP_DSHOW);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera." << std::endl;
        return -1;
    }

    // 创建窗口
    //namedWindow("Camera");

    // 设置分辨率
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    // 获取图像尺寸
    int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    std::cout << "Image size: " << width << "x" << height << std::endl;
    // 创建ORB检测器
    Ptr<ORB> orb = ORB::create();
    // 创建特征匹配器
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //创建提取特征点后的图像
    Mat image_keypoints_left, image_keypoints_right;
    std::vector<KeyPoint> keypoints_left, keypoints_right;
    Mat descriptors_left, descriptors_right;
    // 循环读取并显示帧
    Mat frame;
    while (true) {
        //cap >> frame;
        cap.read(frame);
        Mat roi1 = frame(Rect(0, 0, frame.cols / 2, frame.rows));
        Mat roi2 = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));

        // 检测关键点和计算描述符
        orb->detectAndCompute(roi1, noArray(), keypoints_left, descriptors_left);
        orb->detectAndCompute(roi2, noArray(), keypoints_right, descriptors_right);
        // 在图像中绘制关键点
        drawKeypoints(roi1, keypoints_left, image_keypoints_left, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        drawKeypoints(roi2, keypoints_right, image_keypoints_right, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        //进行左右图像特征匹配
        std::vector<DMatch> matches;
        matcher->match(descriptors_left, descriptors_right, matches);
        //提高匹配准确率
        double max_dist = 0;
        double min_dist = 100;
        for (int i = 0; i < descriptors_left.rows; i++)
        {
            double dist = matches[i].distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }
        std::vector<DMatch> good_matches;
        for (int i = 0; i < descriptors_left.rows; i++)
        {
            if (matches[i].distance <= std::max(2 * min_dist, 0.005))
            {
                good_matches.push_back(matches[i]);
            }
        }
        // 绘制匹配结果
        Mat img_matches;
        drawMatches(image_keypoints_left, keypoints_left, image_keypoints_right, keypoints_right, good_matches, img_matches);
        //显示匹配后的结果
        imshow("after match", img_matches);

        // 显示分割并进行特征点提取后的帧
        //imshow("Left", image_keypoints_left);
        //imshow("Right", image_keypoints_right);

        //imshow("Camera", frame);
        if (waitKey(30) >= 0) break;
    }

    // 关闭窗口
    destroyAllWindows();

    return 0;
}


//双目摄像头测距  无法实现
//#include <opencv2/opencv.hpp>
//
//using namespace cv;
//
//int main()
//{
//    // 打开左右摄像头
//    // 打开默认摄像头
//    VideoCapture cap(1+CAP_DSHOW);
//    if (!cap.isOpened()) {
//        std::cerr << "Failed to open camera." << std::endl;
//        return -1;
//    }
//
//    // 创建窗口
//    //namedWindow("Camera");
//
//    // 设置分辨率
//    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
//    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
//
//    // 创建StereoBM对象
//    Ptr<StereoBM> bm = StereoBM::create(16, 9);
//
//    // 创建Mat对象
//    Mat img_left, img_right, img_disp, img_depth;
//
//    while (true)
//    {
//        Mat frame;
//        // 读取左右摄像头的图像
//        cap.read(frame);
//        Mat roi1 = frame(Rect(0, 0, frame.cols / 2, frame.rows));
//        Mat roi2 = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));
//
//        // 转换为灰度图像
//        cvtColor(roi1, roi1, COLOR_BGR2GRAY);
//        cvtColor(roi2, roi2, COLOR_BGR2GRAY);
//
//        // 计算视差图像
//        bm->compute(roi1, roi2, img_disp);
//
//        // 计算深度图像
//        //img_depth = 1.0 * 16 / img_disp;
//        // 计算深度图像
//        double baseline = 0.1;  // 双目摄像头基线长度
//        double focal_length = 500;  // 摄像头焦距
//        img_depth = baseline * focal_length / img_disp;
//
//        // 显示图像
//        imshow("Left camera", roi1);
//        imshow("Right camera", roi2);
//        imshow("Disparity map", img_disp);
//        imshow("Depth map", img_depth);
//
//        // 按下ESC键退出循环
//        if (waitKey(1) == 27) break;
//    }
//
//    // 释放摄像头
//    cap.release();
//
//    return 0;
//}

//双目摄像头测距另一种方法  可能需要对摄像头进行校准，生成校正文件(待实现)
//#include <opencv2/opencv.hpp>
//
//using namespace cv;
//
//int main()
//{
//    // 打开左右摄像头
//    // 打开默认摄像头
//    VideoCapture cap(1+CAP_DSHOW);
//    if (!cap.isOpened()) {
//        std::cerr << "Failed to open camera." << std::endl;
//        return -1;
//    }
//    // 设置分辨率
//    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
//    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
//
//    // 标定参数
//    Mat cameraMatrix_left, distCoeffs_left;
//    Mat cameraMatrix_right, distCoeffs_right;
//    Mat R, T, E, F;
//    Size imageSize(640, 480);
//
//    //生成一个这个文件
//    FileStorage fs("calibration.yml", FileStorage::READ);
//
//    fs["camera_matrix_left"] >> cameraMatrix_left;
//    fs["distortion_coefficients_left"] >> distCoeffs_left;
//    fs["camera_matrix_right"] >> cameraMatrix_right;
//    fs["distortion_coefficients_right"] >> distCoeffs_right;
//    fs["R"] >> R;
//    fs["T"] >> T;
//    fs.release();
//
//    // 创建双目摄像头对象
//    Ptr<StereoBM> stereo = StereoBM::create();
//
//    while (true)
//    {
//        Mat frame, roi1, roi2;
//        cap >> frame;
//
//        if (frame.empty())
//            break;
//        roi1 = frame(Rect(0, 0, frame.cols / 2, frame.rows));
//        roi2 = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));
//        // 图像矫正
//        Mat R1, R2, P1, P2, Q;
//        stereoRectify(cameraMatrix_left, distCoeffs_left, cameraMatrix_right, distCoeffs_right, imageSize, R, T, R1, R2, P1, P2, Q);
//        Mat map1x, map1y, map2x, map2y;
//        initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, imageSize, CV_32FC1, map1x, map1y);
//        initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, imageSize, CV_32FC1, map2x, map2y);
//        Mat frame_left_rect, frame_right_rect;
//        remap(roi1, frame_left_rect, map1x, map1y, INTER_LINEAR);
//        remap(roi2, frame_right_rect, map2x, map2y, INTER_LINEAR);
//
//        // 计算视差
//        Mat disp;
//        stereo->compute(frame_left_rect, frame_right_rect, disp);
//
//        // 计算深度
//        double baseline = norm(T);
//        double focalLength = cameraMatrix_left.at<double>(0, 0);
//        Mat depth;
//        depth = baseline * focalLength / disp;
//
//        // 显示深度图
//        imshow("Depth", depth);
//
//        char key = waitKey(1);
//        if (key == 27)
//            break;
//    }
//
//    cap.release();
//    destroyAllWindows();
//    return 0;
//}


//特征点跟踪
//#include <opencv2/opencv.hpp>
//
//using namespace cv;
//
//int main(int argc, char* argv[]) {
//    // 打开摄像头
//    VideoCapture cap(0);
//    if (!cap.isOpened()) {
//        std::cerr << "Cannot open camera!" << std::endl;
//        return -1;
//    }
//
//    // 创建ORB特征检测器和描述符提取器
//    Ptr<ORB> orb = ORB::create();
//
//    // 创建BFMatcher匹配器
//    BFMatcher matcher(NORM_HAMMING);
//
//    // 创建窗口
//    namedWindow("Feature tracking", WINDOW_NORMAL);
//
//    // 跟踪特征点
//    std::vector<KeyPoint> keypoints;
//    Mat descriptors;
//    Mat prev_frame, curr_frame;
//    std::vector<Point2f> prev_points, curr_points;
//    std::vector<uchar> status;
//    std::vector<float> error;
//    while (true) {
//        // 读取当前帧
//        cap >> curr_frame;
//        resize(curr_frame, curr_frame, Size(0,0), 0.5, 0.5);
//        if (curr_frame.empty()) {
//            break;
//        }
//
//        // 提取当前帧的ORB特征点和描述符
//        orb->detectAndCompute(curr_frame, Mat(), keypoints, descriptors);
//
//        // 如果是第一帧，则保存特征点和描述符
//        if (prev_frame.empty()) {
//            prev_frame = curr_frame.clone();
//            prev_points.clear();
//            for (const auto& kp : keypoints) {
//                prev_points.push_back(kp.pt);
//            }
//            descriptors.copyTo(descriptors);
//            continue;
//        }
//
//        // 使用BFMatcher匹配器进行特征点匹配
//        std::vector<DMatch> matches;
//        matcher.match(descriptors, descriptors, matches);
//
//        // 提取匹配成功的特征点
//        curr_points.clear();
//        for (const auto& match : matches) {
//            curr_points.push_back(keypoints[match.trainIdx].pt);
//        }
//
//        // 使用光流法计算特征点的运动向量
//        calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_points, curr_points, status, error);
//
//        // 绘制特征点和运动向量
//        for (size_t i = 0; i < curr_points.size(); i++) {
//            if (status[i]) {
//                line(curr_frame, prev_points[i], curr_points[i], Scalar(0, 255, 0));
//                circle(curr_frame, curr_points[i], 2, Scalar(0, 0, 255), -1);
//            }
//        }
//
//        // 更新前一帧的特征点和描述符
//        prev_frame = curr_frame.clone();
//        prev_points = curr_points;
//        descriptors.copyTo(descriptors);
//
//        // 显示处理结果
//        imshow("Feature tracking", curr_frame);
//
//        // 等待按下ESC键退出程序
//        if (waitKey(1) == 27) {
//            break;
//        }
//    }
//
//    return 0;
//}
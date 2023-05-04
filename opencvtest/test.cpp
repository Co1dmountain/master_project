

//�������⼰ƥ��  �ɹ�ʵ��
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    // ��Ĭ������ͷ
    VideoCapture cap(1+CAP_DSHOW);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera." << std::endl;
        return -1;
    }

    // ��������
    //namedWindow("Camera");

    // ���÷ֱ���
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    // ��ȡͼ��ߴ�
    int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    std::cout << "Image size: " << width << "x" << height << std::endl;
    // ����ORB�����
    Ptr<ORB> orb = ORB::create();
    // ��������ƥ����
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //������ȡ��������ͼ��
    Mat image_keypoints_left, image_keypoints_right;
    std::vector<KeyPoint> keypoints_left, keypoints_right;
    Mat descriptors_left, descriptors_right;
    // ѭ����ȡ����ʾ֡
    Mat frame;
    while (true) {
        //cap >> frame;
        cap.read(frame);
        Mat roi1 = frame(Rect(0, 0, frame.cols / 2, frame.rows));
        Mat roi2 = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));

        // ���ؼ���ͼ���������
        orb->detectAndCompute(roi1, noArray(), keypoints_left, descriptors_left);
        orb->detectAndCompute(roi2, noArray(), keypoints_right, descriptors_right);
        // ��ͼ���л��ƹؼ���
        drawKeypoints(roi1, keypoints_left, image_keypoints_left, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        drawKeypoints(roi2, keypoints_right, image_keypoints_right, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        //��������ͼ������ƥ��
        std::vector<DMatch> matches;
        matcher->match(descriptors_left, descriptors_right, matches);
        //���ƥ��׼ȷ��
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
        // ����ƥ����
        Mat img_matches;
        drawMatches(image_keypoints_left, keypoints_left, image_keypoints_right, keypoints_right, good_matches, img_matches);
        //��ʾƥ���Ľ��
        imshow("after match", img_matches);

        // ��ʾ�ָ������������ȡ���֡
        //imshow("Left", image_keypoints_left);
        //imshow("Right", image_keypoints_right);

        //imshow("Camera", frame);
        if (waitKey(30) >= 0) break;
    }

    // �رմ���
    destroyAllWindows();

    return 0;
}


//˫Ŀ����ͷ���  �޷�ʵ��
//#include <opencv2/opencv.hpp>
//
//using namespace cv;
//
//int main()
//{
//    // ����������ͷ
//    // ��Ĭ������ͷ
//    VideoCapture cap(1+CAP_DSHOW);
//    if (!cap.isOpened()) {
//        std::cerr << "Failed to open camera." << std::endl;
//        return -1;
//    }
//
//    // ��������
//    //namedWindow("Camera");
//
//    // ���÷ֱ���
//    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
//    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
//
//    // ����StereoBM����
//    Ptr<StereoBM> bm = StereoBM::create(16, 9);
//
//    // ����Mat����
//    Mat img_left, img_right, img_disp, img_depth;
//
//    while (true)
//    {
//        Mat frame;
//        // ��ȡ��������ͷ��ͼ��
//        cap.read(frame);
//        Mat roi1 = frame(Rect(0, 0, frame.cols / 2, frame.rows));
//        Mat roi2 = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));
//
//        // ת��Ϊ�Ҷ�ͼ��
//        cvtColor(roi1, roi1, COLOR_BGR2GRAY);
//        cvtColor(roi2, roi2, COLOR_BGR2GRAY);
//
//        // �����Ӳ�ͼ��
//        bm->compute(roi1, roi2, img_disp);
//
//        // �������ͼ��
//        //img_depth = 1.0 * 16 / img_disp;
//        // �������ͼ��
//        double baseline = 0.1;  // ˫Ŀ����ͷ���߳���
//        double focal_length = 500;  // ����ͷ����
//        img_depth = baseline * focal_length / img_disp;
//
//        // ��ʾͼ��
//        imshow("Left camera", roi1);
//        imshow("Right camera", roi2);
//        imshow("Disparity map", img_disp);
//        imshow("Depth map", img_depth);
//
//        // ����ESC���˳�ѭ��
//        if (waitKey(1) == 27) break;
//    }
//
//    // �ͷ�����ͷ
//    cap.release();
//
//    return 0;
//}

//˫Ŀ����ͷ�����һ�ַ���  ������Ҫ������ͷ����У׼������У���ļ�(��ʵ��)
//#include <opencv2/opencv.hpp>
//
//using namespace cv;
//
//int main()
//{
//    // ����������ͷ
//    // ��Ĭ������ͷ
//    VideoCapture cap(1+CAP_DSHOW);
//    if (!cap.isOpened()) {
//        std::cerr << "Failed to open camera." << std::endl;
//        return -1;
//    }
//    // ���÷ֱ���
//    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
//    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
//
//    // �궨����
//    Mat cameraMatrix_left, distCoeffs_left;
//    Mat cameraMatrix_right, distCoeffs_right;
//    Mat R, T, E, F;
//    Size imageSize(640, 480);
//
//    //����һ������ļ�
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
//    // ����˫Ŀ����ͷ����
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
//        // ͼ�����
//        Mat R1, R2, P1, P2, Q;
//        stereoRectify(cameraMatrix_left, distCoeffs_left, cameraMatrix_right, distCoeffs_right, imageSize, R, T, R1, R2, P1, P2, Q);
//        Mat map1x, map1y, map2x, map2y;
//        initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, imageSize, CV_32FC1, map1x, map1y);
//        initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, imageSize, CV_32FC1, map2x, map2y);
//        Mat frame_left_rect, frame_right_rect;
//        remap(roi1, frame_left_rect, map1x, map1y, INTER_LINEAR);
//        remap(roi2, frame_right_rect, map2x, map2y, INTER_LINEAR);
//
//        // �����Ӳ�
//        Mat disp;
//        stereo->compute(frame_left_rect, frame_right_rect, disp);
//
//        // �������
//        double baseline = norm(T);
//        double focalLength = cameraMatrix_left.at<double>(0, 0);
//        Mat depth;
//        depth = baseline * focalLength / disp;
//
//        // ��ʾ���ͼ
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


//���������
//#include <opencv2/opencv.hpp>
//
//using namespace cv;
//
//int main(int argc, char* argv[]) {
//    // ������ͷ
//    VideoCapture cap(0);
//    if (!cap.isOpened()) {
//        std::cerr << "Cannot open camera!" << std::endl;
//        return -1;
//    }
//
//    // ����ORB�������������������ȡ��
//    Ptr<ORB> orb = ORB::create();
//
//    // ����BFMatcherƥ����
//    BFMatcher matcher(NORM_HAMMING);
//
//    // ��������
//    namedWindow("Feature tracking", WINDOW_NORMAL);
//
//    // ����������
//    std::vector<KeyPoint> keypoints;
//    Mat descriptors;
//    Mat prev_frame, curr_frame;
//    std::vector<Point2f> prev_points, curr_points;
//    std::vector<uchar> status;
//    std::vector<float> error;
//    while (true) {
//        // ��ȡ��ǰ֡
//        cap >> curr_frame;
//        resize(curr_frame, curr_frame, Size(0,0), 0.5, 0.5);
//        if (curr_frame.empty()) {
//            break;
//        }
//
//        // ��ȡ��ǰ֡��ORB�������������
//        orb->detectAndCompute(curr_frame, Mat(), keypoints, descriptors);
//
//        // ����ǵ�һ֡���򱣴��������������
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
//        // ʹ��BFMatcherƥ��������������ƥ��
//        std::vector<DMatch> matches;
//        matcher.match(descriptors, descriptors, matches);
//
//        // ��ȡƥ��ɹ���������
//        curr_points.clear();
//        for (const auto& match : matches) {
//            curr_points.push_back(keypoints[match.trainIdx].pt);
//        }
//
//        // ʹ�ù�����������������˶�����
//        calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_points, curr_points, status, error);
//
//        // ������������˶�����
//        for (size_t i = 0; i < curr_points.size(); i++) {
//            if (status[i]) {
//                line(curr_frame, prev_points[i], curr_points[i], Scalar(0, 255, 0));
//                circle(curr_frame, curr_points[i], 2, Scalar(0, 0, 255), -1);
//            }
//        }
//
//        // ����ǰһ֡���������������
//        prev_frame = curr_frame.clone();
//        prev_points = curr_points;
//        descriptors.copyTo(descriptors);
//
//        // ��ʾ������
//        imshow("Feature tracking", curr_frame);
//
//        // �ȴ�����ESC���˳�����
//        if (waitKey(1) == 27) {
//            break;
//        }
//    }
//
//    return 0;
//}
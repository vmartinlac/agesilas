#include <map>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

class UnionFind
{
public:

    void init(int count)
    {
        myParents.resize(count);

        for(int i=0; i<count; i++)
        {
            myParents[i] = i;
        }
    }

    void union_(int a, int b)
    {
        myParents[find(b)] = find(a);
    }

    int find(int a)
    {
        while(myParents[myParents[a]] != myParents[a])
        {
            myParents[a] = myParents[myParents[a]];
        }

        return myParents[a];
    }

    void build(std::vector<int>& classes, int& num_classes)
    {
        classes.assign(myParents.size(), -1);

        num_classes = 0;

        for(int i=0; i<myParents.size(); i++)
        {
            int j = find(i);

            if(classes[j] == -1)
            {
                classes[j] = num_classes;
                classes[i] = num_classes;
                num_classes++;
            }
            else
            {
                classes[i] = classes[j];
            }
        }
    }

protected:

    std::vector<int> myParents;
};

void detect_corners(const cv::Mat3b& image, std::vector<cv::Point2f>& corners)
{
    const cv::Size thumbnail_size(640, 480);

    corners.clear();

    cv::Mat1b gray;

    double gamma = 1.0;

    if(image.cols < thumbnail_size.width && image.rows < thumbnail_size.height)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        gamma = 1.0;
    }
    else
    {
        const double gammax = double(thumbnail_size.width) / double(image.cols);
        const double gammay = double(thumbnail_size.height) / double(image.rows);
        gamma = std::min<double>(gammax, gammay);

        if( 1.0 < 3.0*gamma*gamma )
        {
            cv::Mat1b tmp;
            cv::cvtColor(image, tmp, cv::COLOR_BGR2GRAY);
            cv::resize(tmp, gray, cv::Size(0, 0), gamma, gamma, cv::INTER_LINEAR);
        }
        else
        {
            cv::Mat3b tmp;
            cv::resize(image, tmp, cv::Size(0, 0), gamma, gamma, cv::INTER_LINEAR);
            cv::cvtColor(tmp, gray, cv::COLOR_BGR2GRAY);
        }
    }

    cv::threshold(gray, gray, 0.0, 255.0, cv::THRESH_OTSU);

    UnionFind uf;
    uf.init(gray.cols*gray.rows);

    for(int i=0; i<gray.rows-1; i++)
    {
        for(int j=0; j<gray.cols; j++)
        {
            if(gray(i,j) == gray(i+1,j))
            {
                uf.union_( i*gray.cols+j, (i+1)*gray.cols+j );
            }
        }
    }

    for(int i=0; i<gray.rows; i++)
    {
        for(int j=0; j<gray.cols-1; j++)
        {
            if(gray(i,j) == gray(i,j+1))
            {
                uf.union_( i*gray.cols+j, i*gray.cols+j+1 );
            }
        }
    }

    int num_classes = 0;
    std::vector<int> classes;
    uf.build(classes, num_classes);

    bool ok = (num_classes == 15);

    if(ok)
    {
        std::vector<int> num_neighbors(num_classes, 0);
        std::vector<bool> neighbors;
        neighbors.assign(num_classes*num_classes, false);

        for(int i=0; i<gray.rows-1; i++)
        {
            for(int j=0; j<gray.cols; j++)
            {
                const int c0 = classes[i*gray.cols+j];
                const int c1 = classes[(i+1)*gray.cols+j];

                if(c0 != c1)
                {
                    neighbors[15*c0+c1] = true;
                    neighbors[15*c1+c0] = true;
                }
            }
        }

        for(int i=0; i<gray.rows; i++)
        {
            for(int j=0; j<gray.cols-1; j++)
            {
                const int c0 = classes[i*gray.cols+j];
                const int c1 = classes[i*gray.cols+j+1];

                if(c0 != c1)
                {
                    neighbors[15*c0+c1] = true;
                    neighbors[15*c1+c0] = true;
                }
            }
        }

        int num13 = 0;
        int num2 = 0;
        int num1 = 0;

        int square_class = -1;
        int background_class = -1;

        for(int i=0; ok && i<num_classes; i++)
        {
            int count = 0;

            for(int j=0; j<num_classes; j++)
            {
                if(neighbors[i*15+j])
                {
                    count++;
                }
            }

            num_neighbors[i] = count;

            if(count == 2)
            {
                num2++;
                square_class = i;
            }
            else if(count == 1)
            {
                num1++;
            }
            else if(count == 13)
            {
                num13++;
                background_class = i;
            }
            else
            {
                ok = false;
            }
        }

        if(ok)
        {
            ok = (num13 == 1 && num2 == 1 && num1 == 13 && square_class >= 0 && background_class >= 0);
        }

        if(ok)
        {
            std::vector<int> corner_indices;
            corner_indices.assign(num_classes, -1);
            int num_corners_found = 0;

            bool already_found = false;

            for(int i=0; ok && i<num_classes; i++)
            {
                if(neighbors[15*i+background_class])
                {
                    if(num_neighbors[i] == 1)
                    {
                        corner_indices[i] = num_corners_found;
                        num_corners_found++;
                    }
                    else if(num_neighbors[i] == 2)
                    {
                        if(already_found)
                        {
                            ok = false;
                        }
                        else
                        {
                            already_found = true;
                        }
                    }
                    else
                    {
                        ok = false;
                    }
                }
            }

            if(ok)
            {
                ok = (already_found && num_corners_found == 12);
            }

            if(ok)
            {
                std::vector<int> num_pixels;
                num_pixels.assign(12, 0);

                corners.assign(12, cv::Point2f(0.0f, 0.0f));

                for(int i=0; i<gray.rows; i++)
                {
                    for(int j=0; j<gray.cols; j++)
                    {
                        const int corner = corner_indices[classes[i*gray.cols+j]];

                        if(corner >= 0)
                        {
                            corners[corner] += cv::Point2f(double(j)/gamma, double(i)/gamma);
                            num_pixels[corner]++;
                        }
                    }
                }

                for(int i=0; ok && i<corners.size(); i++)
                {
                    if(num_pixels[i] > 0)
                    {
                        corners[i] /= num_pixels[i];
                    }
                    else
                    {
                        ok = false;
                    }
                }
            }
        }
    }

    if(!ok)
    {
        corners.clear();
    }

    cv::imwrite("im0.png", gray);
}

void track_corners(const cv::Mat3b& image, std::vector<cv::Point2f>& corners)
{
}

int main(int num_args, char** args)
{
    std::string fname = "../IMG_20210102_151218.jpg";
    if(num_args == 2)
    {
        fname = args[1];
    }
    cv::Mat3b img0 = cv::imread(fname);
    if(img0.data == nullptr) throw std::runtime_error("Could not load image");

    std::vector<cv::Point2f> corners;
    detect_corners(img0, corners);

    for(const cv::Point2f& x : corners)
    {
        cv::circle(img0, x, 10, cv::Vec3b(0, 255, 0), -1);
    }

    cv::imwrite("im1.png", img0);
    return 0;
}


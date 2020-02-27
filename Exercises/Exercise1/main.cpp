/**
 * Course: 3D Scanning and Motion Capture
 * File: main.cpp
 * Purpose: First exercise of the course.
 * @author Juan Raul Padron Griffe, Wojciech Zielonka
 * @version 1.0 26/10/2018
*/


#include <iostream>
#include <fstream>
#include <vector>

#include "Eigen.h"

#include "VirtualSensor.h"

struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // position stored as 4 floats (4th component is supposed to be 1.0)
    Vector4f position;
    // color stored as 4 unsigned char
    Vector4uc color;
};

//A triangle is valid to save when its vertices and edges are valid
inline bool ValidTriangle(Vector4f p0, Vector4f p1, Vector4f p2, float edgeThreshold) {

    //Valid vertex
    if (p0.x() == MINF || p1.x() == MINF || p2.x() == MINF) return false;

    //Valid edges: distance of the edges must be smaller than the edge threshold
    return !((p0 - p1).norm() >= edgeThreshold || (p1 - p2).norm() >= edgeThreshold ||
             (p2 - p1).norm() >= edgeThreshold);

}

bool WriteMesh(Vertex *vertices, unsigned int width, unsigned int height, const std::string &filename) {
    float edgeThreshold = 0.01f; // 1cm

    // TODO 2: use the OFF file format to save the vertices grid (http://www.geomview.org/docs/html/OFF.html)
    // - have a look at the "off_sample.off" file to see how to store the vertices and triangles
    // - for debugging we recommend to first only write out the vertices (set the number of faces to zero)
    // - for simplicity write every vertex to file, even if it is not valid (position.x() == MINF) (note that all vertices in the off file have to be valid, thus, if a point is not valid write out a dummy point like (0,0,0))
    // - use a simple triangulation exploiting the grid structure (neighboring vertices build a triangle, two triangles per grid cell)
    // - you can use an arbitrary triangulation of the cells, but make sure that the triangles are consistently oriented
    // - only write triangles with valid vertices and an edge length smaller then edgeThreshold

    // TODO: Get number of vertices
    unsigned int nVertices = height * width;

    // TODO: Get number of faces
    //unsigned nFaces = 2 * (height-1)* (width-1);//Without discarding faces

    //Building mesh (Slide 16 of the exercise).
    std::vector<Vector3i> faces;
    unsigned int idx0, idx1, idx2, idx3;
    Vector4f p0, p1, p2, p3;

    for (int y = 0; y < height - 1; y++) {
        for (int x = 0; x < width - 1; x++) {

            //Indices (Ensure consistent orientation of the triangles)
            idx0 = y * width + x;
            idx1 = idx0 + 1;
            idx2 = idx0 + width;
            idx3 = idx2 + 1;

            //Points
            p0 = vertices[idx0].position;
            p1 = vertices[idx1].position;
            p2 = vertices[idx2].position;
            p3 = vertices[idx3].position;

            //Upper Triangle
            if (ValidTriangle(p0, p2, p1, edgeThreshold)) {
                faces.emplace_back(idx0, idx2, idx1);
            }

            //Bottom Triangle
            if (ValidTriangle(p2, p3, p1, edgeThreshold)) {
                faces.emplace_back(idx2, idx3, idx1);
            }
        }
    }

    unsigned nFaces = faces.size();

    // Write off file
    std::ofstream outFile(filename);
    if (!outFile.is_open()) return false;

    // write header
    outFile << "COFF" << std::endl;
    outFile << nVertices << " " << nFaces << " 0" << std::endl;

    // TODO: save vertices
    outFile << "# list of vertices" << std::endl;
    outFile << "# X Y Z R G B A" << std::endl;

    for (int i = 0; i < nVertices; i++) {

        auto position = (vertices[i].position.x() == MINF) ? Vector4f(0.0, 0.0, 0.0, 1.0) : vertices[i].position;
        auto color = vertices[i].color;

        //std::cout  << position[0] << " " << position[1] << " " << position[2] << std::endl;
        //std::cout  << +color[0] << " " << +color[1] << " " << +color[2] << " " << +color[3] << std::endl;

        outFile << position[0] << " " << position[1] << " " << position[2] << " ";
        outFile << +color[0] << " " << +color[1] << " " << +color[2] << " " << +color[3]
                << std::endl;//+ to force to print as an int
    }

    // TODO: save faces
    outFile << "# list of faces" << std::endl;
    outFile << "# nVerticesPerFace idx0 idx1 idx2 ..." << std::endl;

    for (int i = 0; i < nFaces; i++) {

        auto face = faces[i];

        //std::cout << face[0] << " " << face[1] << " " << face[2] << std::endl;
        outFile << "3 " << face[0] << " " << face[1] << " " << face[2] << std::endl;
    }

    //Release vector memory
    faces.clear();

    // close file
    outFile.close();

    return true;
}

int main() {
    std::string filenameIn = "../data/rgbd_dataset_freiburg1_xyz/";
    std::string filenameBaseOut = "mesh_";

    // load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor;
    if (!sensor.Init(filenameIn)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
        return -1;
    }

    // convert video to meshes
    while (sensor.ProcessNextFrame()) {
        // get ptr to the current depth frame
        // depth is stored in row major (get dimensions via sensor.GetDepthImageWidth() / GetDepthImageHeight())
        float *depthMap = sensor.GetDepth();
        // get ptr to the current color frame
        // color is stored as RGBX in row major (4 byte values per pixel, get dimensions via sensor.GetColorImageWidth() / GetColorImageHeight())
        BYTE *colorMap = sensor.GetColorRGBX();

        // get depth intrinsics
        Matrix3f depthIntrinsics = sensor.GetDepthIntrinsics();
        float fovX = depthIntrinsics(0, 0);
        float fovY = depthIntrinsics(1, 1);
        float cX = depthIntrinsics(0, 2);
        float cY = depthIntrinsics(1, 2);

        // compute inverse depth extrinsics
        Matrix4f depthExtrinsicsInv = sensor.GetDepthExtrinsics().inverse();

        Matrix4f trajectory = sensor.GetTrajectory();
        Matrix4f trajectoryInv = sensor.GetTrajectory().inverse();

        //Inverse of depth camera intrinsics (Slides 12, 25 of the exercise)
        Matrix4f depthIntrinsicsInv;

        depthIntrinsicsInv << 1.0f / fovX, 0.0f, -cX / fovX, 0.0f,
                                0.0f, 1.0f / fovY, -cY / fovY, 0.0f,
                                0.0f, 0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 1.0f;

        // TODO 1: back-projection
        // write result to the vertices array below, keep pixel ordering!
        // if the depth value at idx is invalid (MINF) write the following values to the vertices array
        // vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
        // vertices[idx].color = Vector4uc(0,0,0,0);
        // otherwise apply back-projection and transform the vertex to world space, use the corresponding color from the colormap

        unsigned int width = sensor.GetDepthImageWidth();
        unsigned int height = sensor.GetDepthImageHeight();

        auto vertices = new Vertex[height * width];

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {

                int idx = y * width + x;
                float depth = depthMap[idx];

                if (depth == MINF) {
                    vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
                    vertices[idx].color = Vector4uc(0, 0, 0, 0);
                } else {
                    int colorIdx = idx * 4;

                    Vector4uc color = Vector4uc(colorMap[colorIdx], colorMap[colorIdx + 1], colorMap[colorIdx + 2],
                                                colorMap[colorIdx + 3]);

                    //Transformations based on projection pipeline (Slide 12 of the exercise).

                    //Pixel space -> image space
                    Vector4f image = Vector4f(x * depth, y * depth, depth, 1.0f);

                    //Image space -> camera space
                    Vector4f camera = depthIntrinsicsInv * image;

                    //Camera space to world space
                    Vector4f world = trajectoryInv * depthExtrinsicsInv * camera;

                    vertices[idx].position = world;
                    vertices[idx].color = color;
                }
            }

        // write mesh file
        std::stringstream ss;
        ss << filenameBaseOut << sensor.GetCurrentFrameCnt() << ".off";
        if (!WriteMesh(vertices, width, height, ss.str())) {
            std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
            return -1;
        }

        // free mem
        delete[] vertices;
    }

    return 0;
}

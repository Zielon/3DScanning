#include <iostream>
#include <fstream>

#include "Eigen.h"

#include "VirtualSensor.h"

struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	// position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// color stored as 4 unsigned char
	Vector4uc color;
};

bool WriteMesh(Vertex* vertices, unsigned int width, unsigned int height, const std::string& filename)
{
	float edgeThreshold = 0.01f; // 1cm

	// TODO 2: use the OFF file format to save the vertices grid (http://www.geomview.org/docs/html/OFF.html)
	// - have a look at the "off_sample.off" file to see how to store the vertices and triangles
	// - for debugging we recommend to first only write out the vertices (set the number of faces to zero)
	// - for simplicity write every vertex to file, even if it is not valid (position.x() == MINF) (note that all vertices in the off file have to be valid, thus, if a point is not valid write out a dummy point like (0,0,0))
	// - use a simple triangulation exploiting the grid structure (neighboring vertices build a triangle, two triangles per grid cell)
	// - you can use an arbitrary triangulation of the cells, but make sure that the triangles are consistently oriented
	// - only write triangles with valid vertices and an edge length smaller then edgeThreshold

	// TODO: Get number of vertices
	unsigned int nVertices = height*width;

	// TODO: Get number of faces
	unsigned nFaces = 2 * (height-1)* (width-1);
    //unsigned nFaces = 0;

	// Write off file
	std::ofstream outFile(filename);
	if (!outFile.is_open()) return false;

	// write header
	outFile << "COFF" << std::endl;
	outFile << nVertices << " " << nFaces << " 0" << std::endl;

	// TODO: save vertices

    outFile << "# list of vertices" << std::endl;
    outFile << "# X Y Z R G B A" << std::endl;

    for (int i = 0; i < nVertices; i++){

        auto position = (vertices[i].position.x() == MINF)? Vector4f(0.0, 0.0, 0.0, 0.0): vertices[i].position;
        auto color = vertices[i].color;

        //std::cout  << position[0] << " " << position[1] << " " << position[2] << std::endl;
        //std::cout  << +color[0] << " " << +color[1] << " " << +color[2] << " " << +color[3] << std::endl;

        outFile << position[0] << " " << position[1] << " " << position[2] << " ";
        outFile << +color[0] << " " << +color[1] << " " << +color[2] << " " << +color[3] << std::endl;//+ to force to print as a int
    }

	// TODO: save faces
    outFile << "# list of faces" << std::endl;
    outFile << "# nVerticesPerFace idx0 idx1 idx2 ..." << std::endl;

    for(int y = 0; y < height-1; y++)
        for(int x = 0; x < width-1; x++){

            int idx = y * width + x;
            int idx2 = idx+width; //Next row

            //Upper Triangle
            outFile << idx << " " << idx2 << " " << idx+1 << std::endl;

            //Bottom Triangle
            outFile << idx2 << " " << idx2+1 << " " << idx+1 << std::endl;
        }


	// close file
	outFile.close();

	return true;
}

int main()
{
	std::string filenameIn = "../data/rgbd_dataset_freiburg1_xyz/";
	std::string filenameBaseOut = "mesh_";

	// load video
	std::cout << "Initialize virtual sensor..." << std::endl;
	VirtualSensor sensor;
	if (!sensor.Init(filenameIn))
	{
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return -1;
	}

	// convert video to meshes
	while (sensor.ProcessNextFrame())
	{
		// get ptr to the current depth frame
		// depth is stored in row major (get dimensions via sensor.GetDepthImageWidth() / GetDepthImageHeight())
		float* depthMap = sensor.GetDepth();
		// get ptr to the current color frame
		// color is stored as RGBX in row major (4 byte values per pixel, get dimensions via sensor.GetColorImageWidth() / GetColorImageHeight())
		BYTE* colorMap = sensor.GetColorRGBX();

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

        // compute inverse depth intrinsics
        MatrixXf depthIntrinsicsInv(4,3);

        depthIntrinsicsInv << 1.0f/fovX, 0.0f , -cX/fovX,
                0.0f , 1.0f/fovY, -cY/fovY,
                0.0f , 0.0f , 1.0f ,
                0.0f, 0.0f, 0.0f;

        //MatrixXf depthIntrinsicsInv = sensor.GetDepthIntrinsics().inverse();

        std::cout << depthIntrinsicsInv << std::endl << std::endl;

		// TODO 1: back-projection
		// write result to the vertices array below, keep pixel ordering!
		// if the depth value at idx is invalid (MINF) write the following values to the vertices array
		// vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
		// vertices[idx].color = Vector4uc(0,0,0,0);
		// otherwise apply back-projection and transform the vertex to world space, use the corresponding color from the colormap

		unsigned int width = sensor.GetDepthImageWidth();
		unsigned int height = sensor.GetDepthImageHeight();

		auto vertices = new Vertex[height * width];

		for(int y = 0; y < height; y++)
		    for(int x = 0; x < width; x++){

		        int idx = y * width + x;
                float depth = depthMap[idx];

                if(depth == MINF){
                    vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
                    vertices[idx].color = Vector4uc(0,0,0,0);
                }else{
                	int colorIdx = idx * 4;

                    Vector4uc color = Vector4uc(colorMap[colorIdx], colorMap[colorIdx + 1], colorMap[colorIdx + 2], colorMap[colorIdx + 3]);

                    Vector3f image = Vector3f(x*depth, y*depth, depth);
                    Vector4f camera = depthIntrinsicsInv * image;
                    Vector4f world = trajectoryInv * depthExtrinsicsInv * camera;

                    //Vector4f screen = Vector4f(x, y, 0, 0);
                    //Vector4f world = screen.transpose() * depthExtrinsicsInv * trajectoryInv;

					vertices[idx].position = world;
					vertices[idx].color = color;
                }

                //std::cout << vertices[idx].position << std::endl;
                //std::cout << vertices[idx].color << std::endl;
        }

		// write mesh file
		std::stringstream ss;
		ss << filenameBaseOut << sensor.GetCurrentFrameCnt() << ".off";
		if (!WriteMesh(vertices, sensor.GetDepthImageWidth(), sensor.GetDepthImageHeight(), ss.str()))
		{
			std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
			return -1;
		}

		// free mem
		delete[] vertices;
	}

	return 0;
}

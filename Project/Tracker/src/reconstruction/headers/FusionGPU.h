#include "../headers/FusionBase.h"


#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>

#include <cwchar>

#pragma comment (lib, "d3d11.lib")
#include <d3d11.h>

#pragma comment (lib, "D3DCompiler.lib")
#include <D3DCompiler.h>

#define SafeRelease(p) { if ( (p) ) { (p)->Release(); (p) = 0; } }

#define FUSION_SHADER_PATH "./shaders/Fusion.hlsl"


#define FUSION_THREADS 4 
#define MC_THREADS 4
#define MAX_TRIANGLES 4e6

//#define USE_CPU_MC


class FusionGPU : public FusionBase
{
public:
	FusionGPU(SystemParameters camera_parameters, std::string shaderPath = FUSION_SHADER_PATH);
	~FusionGPU();

	//Perfect inheritance ! ! !
	void consume() override {}

	void produce(std::shared_ptr<PointCloud> cloud) override { integrate(cloud);  }

	void wait() const override {}

	// actual stuff

	void integrate(std::shared_ptr<PointCloud> cloud) override;

	void save(std::string name) override;

	void processMesh(Mesh& mesh) override;

private: 
	inline void initWindow();
	inline void initDx11(); 
	inline void initBuffers();
	void reloadShaders(std::string shaderPath);

	inline void populateSettingsBuffers(); 

	inline void initialize(); 

	inline void processMeshCPU(Mesh& mesh); 

	__declspec(align(16)) struct FusionSettings
	{

		Vector3f m_min;
		float m_truncation;
		Vector3f m_max;
		float m_voxel_size;
		float m_focal_length_X = 0;
		float m_focal_length_Y = 0;
		float m_cX = 0;
		float m_cY = 0;

		int m_image_width = 0;
		int m_image_height = 0;
		int m_resSQ;
		int m_resolution;


	}m_fusionSettings;

	__declspec(align(16)) struct FusionPerFrame
	{
		Matrix4f cam2world; 
		Matrix4f world2cam; 
		Vector3i frustum_min;
		Vector3i frustum_max;
		Vector3i numThreads; 
	}m_fusionPerFrame;

	__declspec(align(16)) struct MarchingCubesSettings
	{
		float isolevel = 0; 
	}m_marchingCubesSettings;

	__declspec(align(16)) struct MarchingCubesPerFrame
	{
		int dummy; 
	}m_marchingCubesPerFrame;

	struct __Triangle
	{
		Vector3f v0;
		Vector3f v1;
		Vector3f v2;
	};


	HINSTANCE m_hInstance; 
	HWND m_hWindow; 

	ID3D11Buffer* m_buf_vertexBuffer = NULL;


	ID3D11UnorderedAccessView* m_uav_sdf = NULL;
	ID3D11Buffer* m_buf_sdf = NULL;

	//ID3D11Buffer* m_buf_sdf_copy = NULL;

	ID3D11Texture2D* m_t2d_currentFrame = NULL;
	ID3D11ShaderResourceView* m_srv_currentFrame = NULL;


	ID3D11Buffer* m_cbuf_fusionConst = NULL;
	ID3D11Buffer* m_cbuf_fusionPerFrame = NULL;
	ID3D11Buffer* m_cbuf_marchingCubesConst = NULL;
	ID3D11Buffer* m_cbuf_marchingCubesPerFrame = NULL;

	ID3D11ShaderResourceView* m_srv_fusionConst = NULL;
	ID3D11ShaderResourceView* m_srv_fusionPerFrame = NULL;
	ID3D11ShaderResourceView* m_srv_marchingCubesConst = NULL;
	ID3D11ShaderResourceView* m_srv_marchingCubesPerFrame = NULL;

	ID3D11ComputeShader* m_shader_fusion = NULL; 
	ID3D11ComputeShader* m_shader_marchingCubes = NULL; 
	ID3D11ComputeShader* m_shader_marchingCubesAttachNan = NULL;

	ID3DBlob* m_blob_fusionShader = NULL; 
	ID3DBlob* m_blob_marchingCubesShader = NULL; 
	ID3DBlob* m_blob_marchingCubesAttachNan = NULL;


	ID3D11Device* m_d3dDevice = NULL;
	ID3D11DeviceContext* m_d3dContext = NULL;
	IDXGISwapChain* m_swapChain = NULL;

	ID3D11Debug* m_d3dDebug = NULL;




};

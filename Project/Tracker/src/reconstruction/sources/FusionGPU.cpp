#include "..\headers\FusionGPU.h"

FusionGPU::FusionGPU(SystemParameters camera_parameters) : FusionBase(camera_parameters)
{
	initWindow(); 
	initDx11(); 

	initBuffers(); 
	reloadShaders();

	initialize(); 

	populateSettingsBuffers(); 

}


FusionGPU::~FusionGPU()
{

	SafeRelease(m_uav_sdf);
	SafeRelease(m_t2d_currentFrame);
	SafeRelease(m_buf_sdf);

	SafeRelease(m_srv_fusionConst);
	SafeRelease(m_srv_fusionPerFrame);
	SafeRelease(m_srv_marchingCubesConst);
	SafeRelease(m_srv_marchingCubesPerFrame);
	SafeRelease(m_cbuf_fusionConst);
	SafeRelease(m_cbuf_fusionPerFrame);
	SafeRelease(m_cbuf_marchingCubesConst);
	SafeRelease(m_cbuf_marchingCubesPerFrame);

	SafeRelease(m_shader_marchingCubes); 
	SafeRelease(m_shader_fusion); 

	SafeRelease(m_blob_fusionShader); 
	SafeRelease(m_blob_marchingCubesShader); 

	SafeRelease(m_swapChain)
	SafeRelease(m_d3dContext);
	SafeRelease(m_d3dDevice);

#ifdef _DEBUG
	d3dDebug->ReportLiveDeviceObjects(D3D11_RLDO_DETAIL);

#endif //_DEBUG
}

void FusionGPU::initialize()
{
	m_volume = new Volume(Size(-4, -4, -4), Size(4, 4, 4), 256, 1, false);
	m_trunaction = m_volume->m_voxel_size * 2.f;

	m_fusionSettings.m_max = m_volume->m_max.cast<float>(); 
	m_fusionSettings.m_min = m_volume->m_min.cast<float>();
	m_fusionSettings.m_resolution = m_volume->m_size; 
	m_fusionSettings.m_truncation = m_trunaction; 
	m_fusionSettings.m_voxel_size = m_volume->m_voxel_size; 

	m_fusionSettings.m_image_height = m_camera_parameters.m_image_height; 
	m_fusionSettings.m_image_width = m_camera_parameters.m_image_width;
	m_fusionSettings.m_cX = m_camera_parameters.m_cX; 
	m_fusionSettings.m_cY = m_camera_parameters.m_cY;
	m_fusionSettings.m_focal_length_X = m_camera_parameters.m_focal_length_X;
	m_fusionSettings.m_focal_length_Y = m_camera_parameters.m_focal_length_Y;
	m_fusionSettings.m_depth_min = m_camera_parameters.m_depth_min; 
	m_fusionSettings.m_depth_max = m_camera_parameters.m_depth_max;


}

void FusionGPU::integrate(std::shared_ptr<PointCloud> cloud) 
{
	m_fusionPerFrame.cam2world = cloud->m_pose_estimation; 
	m_fusionPerFrame.world2cam = cloud->m_pose_estimation.inverse();
	const auto frustum_box = computeFrustumBounds(m_fusionPerFrame.cam2world, cloud->m_camera_parameters);
	m_fusionPerFrame.frustum_max = frustum_box.m_max; 
	m_fusionPerFrame.frustum_min = frustum_box.m_min; 


	m_fusionPerFrame.numThreads = frustum_box.m_max - frustum_box.m_min;

	m_d3dContext->UpdateSubresource(m_cbuf_fusionPerFrame, 0, NULL, &m_fusionPerFrame, 0, 0);



	ID3D11Buffer* buffers[] =
	{
		m_cbuf_fusionConst,
		m_cbuf_fusionPerFrame
	};
	m_d3dContext->CSSetConstantBuffers(0, 2, buffers); 
	m_d3dContext->CSSetUnorderedAccessViews(1, 0, &m_uav_sdf, 0);
	m_d3dContext->CSSetShader(m_shader_fusion, NULL, 0);
	m_d3dContext->Dispatch(m_fusionPerFrame.numThreads.x(), m_fusionPerFrame.numThreads.y(), m_fusionPerFrame.numThreads.z());

}

void FusionGPU::save(std::string name)
{
	throw "not impl"; 
}

void FusionGPU::processMesh(Mesh & mesh)
{
	throw "not impl"; 
}



void FusionGPU::populateSettingsBuffers()
{
	m_d3dContext->UpdateSubresource(m_cbuf_fusionConst, 0, NULL, &m_fusionSettings, 0, 0); 
	m_d3dContext->UpdateSubresource(m_cbuf_marchingCubesConst, 0, NULL, &m_marchingCubesSettings, 0, 0);
}










/****
*
*	************************************************ D3d INIT *****************************************
*
*/

void FusionGPU::initBuffers()
{
	ID3DBlob* errBlob;
	HRESULT hr;

	D3D11_BUFFER_DESC descSDF = { 0 };
	descSDF.ByteWidth = sizeof(Voxel) * m_fusionSettings.m_resolution * m_fusionSettings.m_resolution *m_fusionSettings.m_resolution;
	descSDF.StructureByteStride = sizeof(Voxel); 
	descSDF.Usage = D3D11_USAGE_DEFAULT; 
	descSDF.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;

	hr = m_d3dDevice->CreateBuffer(&descSDF, NULL, &m_buf_sdf);
	if (FAILED(hr))
	{
		std::cout << "failed to create buffer" << std::endl;
		std::cin.get();
	}

	D3D11_UNORDERED_ACCESS_VIEW_DESC descSDFUAV; 
	descSDFUAV.Buffer.FirstElement = 0; 
	descSDFUAV.Buffer.Flags = 0; 
	descSDFUAV.Buffer.NumElements = m_fusionSettings.m_resolution * m_fusionSettings.m_resolution *m_fusionSettings.m_resolution;
	descSDFUAV.Format = DXGI_FORMAT_R32G32B32_TYPELESS;
	descSDFUAV.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;

	hr = m_d3dDevice->CreateUnorderedAccessView(m_buf_sdf, &descSDFUAV, &m_uav_sdf);
	if (FAILED(hr))
	{
		std::cout << "failed to create buffer" << std::endl;
		std::cin.get();
	}



	D3D11_TEXTURE2D_DESC descCurrentFrameBuffer = { 0 };
	descCurrentFrameBuffer.Width = m_camera_parameters.m_image_width; 
	descCurrentFrameBuffer.Height = m_camera_parameters.m_image_height;
	descCurrentFrameBuffer.MipLevels = 1; 
	descCurrentFrameBuffer.ArraySize = 1; 
	descCurrentFrameBuffer.Format = DXGI_FORMAT_R32_FLOAT; 
	descCurrentFrameBuffer.SampleDesc.Count = 8; 
	descCurrentFrameBuffer.SampleDesc.Quality = 0; 
	descCurrentFrameBuffer.Usage = D3D11_USAGE_DEFAULT; 
	descCurrentFrameBuffer.BindFlags = D3D11_BIND_SHADER_RESOURCE; 
	descCurrentFrameBuffer.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE; 
	
	hr = m_d3dDevice->CreateTexture2D(&descCurrentFrameBuffer, NULL, &m_t2d_currentFrame); 
	if (FAILED(hr))
	{
		std::cout << "failed to create buffer" << std::endl;
		std::cin.get();
	}

	D3D11_BUFFER_DESC descBuffer = { 0 };
	descBuffer.Usage = D3D11_USAGE_DEFAULT;
	descBuffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	descBuffer.ByteWidth = sizeof(FusionGPU::FusionSettings);

	hr = m_d3dDevice->CreateBuffer(&descBuffer, NULL, &m_cbuf_fusionConst);
	if (FAILED(hr))
	{
		std::cout << "failed to create buffer" << std::endl;
		std::cin.get();
	}
	descBuffer.ByteWidth = sizeof(FusionGPU::FusionPerFrame);

	hr = m_d3dDevice->CreateBuffer(&descBuffer, NULL, &m_cbuf_fusionPerFrame);
	if (FAILED(hr))
	{
		std::cout << "failed to create buffer" << std::endl;
		std::cin.get();
	}	
	
	descBuffer.ByteWidth = sizeof(FusionGPU::MarchingCubesSettings);

	hr = m_d3dDevice->CreateBuffer(&descBuffer, NULL, &m_cbuf_marchingCubesConst);
	if (FAILED(hr))
	{
		std::cout << "failed to create buffer" << std::endl;
		std::cin.get();
	}	
	
	descBuffer.ByteWidth = sizeof(FusionGPU::MarchingCubesPerFrame);

	hr = m_d3dDevice->CreateBuffer(&descBuffer, NULL, &m_cbuf_marchingCubesPerFrame);
	if (FAILED(hr))
	{
		std::cout << "failed to create buffer" << std::endl;
		std::cin.get();
	}
	   
	hr = m_d3dDevice->CreateShaderResourceView(m_cbuf_fusionConst, NULL, &m_srv_fusionConst);
	if (FAILED(hr))
	{
		std::cout << "failed to create SRV" << std::endl;
		std::cin.get();
	}
	hr = m_d3dDevice->CreateShaderResourceView(m_cbuf_fusionPerFrame, NULL, &m_srv_fusionPerFrame);
	if (FAILED(hr))
	{
		std::cout << "failed to create SRV" << std::endl;
		std::cin.get();
	}
	hr = m_d3dDevice->CreateShaderResourceView(m_cbuf_marchingCubesConst, NULL, &m_srv_marchingCubesConst);
	if (FAILED(hr))
	{
		std::cout << "failed to create SRV" << std::endl;
		std::cin.get();
	}
	hr = m_d3dDevice->CreateShaderResourceView(m_cbuf_marchingCubesPerFrame, NULL, &m_srv_marchingCubesPerFrame);
	if (FAILED(hr))
	{
		std::cout << "failed to create SRV" << std::endl;
		std::cin.get();
	}

}


void FusionGPU::reloadShaders()
{
	ID3DBlob* errBlob;
	HRESULT hr; 

	SafeRelease(m_shader_marchingCubes);
	SafeRelease(m_shader_fusion);

	SafeRelease(m_blob_fusionShader);
	SafeRelease(m_blob_marchingCubesShader);

	hr = D3DCompileFromFile(FUSION_SHADER_PATH, NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "vs_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_ENABLE_STRICTNESS, NULL, &m_blob_fusionShader, &errBlob);
	if (FAILED(hr))
	{
		std::cout << "failed to compile Fusion shader " << std::endl;
		std::cout << (char*)errBlob->GetBufferPointer() << std::endl;
		std::cin.get();
	}


	hr = D3DCompileFromFile(FUSION_SHADER_PATH, NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "cs_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_ENABLE_STRICTNESS, NULL, &m_blob_marchingCubesShader, &errBlob);
	if (FAILED(hr))
	{
		std::cout << "failed to compile Marching Cubes shader " << std::endl;
		std::cout << (char*)errBlob->GetBufferPointer() << std::endl;
		std::cin.get();
	}
	hr = m_d3dDevice->CreateComputeShader(m_blob_fusionShader->GetBufferPointer(), m_blob_fusionShader->GetBufferSize(), NULL, &m_shader_fusion); 
	if (FAILED(hr))
	{
		std::cout << "failed to load Fusion Shader " << hr << std::endl;
		std::cin.get();
	}
	hr = m_d3dDevice->CreateComputeShader(m_blob_marchingCubesShader->GetBufferPointer(), m_blob_marchingCubesShader->GetBufferSize(), NULL, &m_shader_marchingCubes);
	if (FAILED(hr))
	{
		std::cout << "failed to load Marching Cubes Shader " << hr << std::endl;
		std::cin.get();
	}
}



void FusionGPU::initDx11()
{
	HRESULT result;

	//Multisample AA
	DXGI_SAMPLE_DESC descSampling;
	descSampling.Count = 8;
	descSampling.Quality = 0;

	DXGI_SWAP_CHAIN_DESC descSwapChain;
	ZeroMemory(&descSwapChain, sizeof(DXGI_SWAP_CHAIN_DESC));
	descSwapChain.BufferDesc.Height = m_camera_parameters.m_image_height;
	descSwapChain.BufferDesc.Width = m_camera_parameters.m_image_width;
	descSwapChain.BufferDesc.RefreshRate.Denominator = 1;
	descSwapChain.BufferDesc.RefreshRate.Numerator = 60;
	descSwapChain.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	descSwapChain.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	descSwapChain.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	descSwapChain.BufferCount = 1;
	descSwapChain.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	descSwapChain.OutputWindow = m_hWindow;
	descSwapChain.SampleDesc = descSampling;
	descSwapChain.Windowed = true;		//window / full screen
	descSwapChain.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

#ifdef _DEBUG
	result = D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, D3D11_CREATE_DEVICE_DEBUG, NULL, NULL, D3D11_SDK_VERSION, &descSwapChain, &swapChain, &d3dDevice, NULL, &d3dContext);
	d3dDevice->QueryInterface(__uuidof(ID3D11Debug), (void**)&d3dDebug);

#else
	result = D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, 0, NULL, NULL, D3D11_SDK_VERSION, &descSwapChain, &m_swapChain, &m_d3dDevice, NULL, &m_d3dContext);
#endif // _DEBUG

	if (FAILED(result))
	{
		std::cout << "D3D init failed" << std::endl;
		std::cin.get();
	}



	D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS options;
	result = m_d3dDevice->CheckFeatureSupport(D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS,
		&options,
		sizeof(D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS));

	if (!options.ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x) {
		MessageBox(NULL, "Compute Shaders are not supported on your hardware", "Unsupported HW", MB_OK);
		return;
	}

}

//Callback function in case of window events
LRESULT CALLBACK windowProc(HWND hWindow, UINT message, WPARAM wParam, LPARAM lParam)
{

	if (message == WM_DESTROY)
	{
		// close the application entirely
		PostQuitMessage(0);
		return 0;
	}


	return DefWindowProc(hWindow, message, wParam, lParam);
}


void FusionGPU::initWindow()
{
	//defining window class parameters
	WNDCLASSEX windowInfo;
	ZeroMemory(&windowInfo, sizeof(WNDCLASSEX));

	m_hInstance = GetModuleHandle(NULL); 

	windowInfo.cbSize = sizeof(WNDCLASSEX);
	windowInfo.style = CS_HREDRAW | CS_VREDRAW;
	windowInfo.lpfnWndProc = &windowProc;
	windowInfo.hInstance = m_hInstance;
	windowInfo.hbrBackground = (HBRUSH)COLOR_WINDOW;		//remove for fullscreen rendering
	windowInfo.lpszClassName = "MarkerlessAR";
	windowInfo.hCursor = LoadCursor(NULL, IDC_ARROW);
	windowInfo.hIcon = LoadIcon(NULL, IDI_WINLOGO);
	windowInfo.hIconSm = LoadIcon(NULL, IDI_WINLOGO);

	//registering window class
	if (!RegisterClassEx(&windowInfo))
	{
		MessageBox(NULL, "Failed to register Window class! ", "ERROR!", MB_OK);
		std::exit(101);
	}

	//creating the window the game is going to be rendered to

	m_hWindow = CreateWindowEx(
		NULL,
		windowInfo.lpszClassName,
		"MarkerlessAR",
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT,
		m_camera_parameters.m_image_width, m_camera_parameters.m_image_height, 
		NULL,
		NULL,
		m_hInstance,
		NULL);

	if (!m_hWindow)
	{
		MessageBox(NULL, "Failed to create Window! ", "ERROR!", MB_OK);
		std::exit(102);
	}


	ShowWindow(m_hWindow, false);

	UpdateWindow(m_hWindow);
}

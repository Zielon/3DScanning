#include "../headers/FusionGPU.h"
#include "../headers/MarchingCubes.h"

FusionGPU::FusionGPU(SystemParameters system_parameters, std::string shaderPath) : FusionBase(system_parameters){

	initialize();

	initWindow();
	initDx11();

	initBuffers();
	reloadShaders(shaderPath);

	populateSettingsBuffers();
}

FusionGPU::~FusionGPU(){

	m_d3dContext->ClearState();
	m_d3dContext->Flush();

	SafeRelease(m_buf_vertexBuffer);
	SafeRelease(m_srv_currentFrame);
	SafeRelease(m_uav_sdf);
	SafeRelease(m_t2d_currentFrame);
	//SafeRelease(m_buf_sdf_copy); 
	SafeRelease(m_buf_sdf);

	SafeRelease(m_srv_fusionConst);
	SafeRelease(m_srv_fusionPerFrame);
	SafeRelease(m_srv_marchingCubesConst);
	SafeRelease(m_srv_marchingCubesPerFrame);
	SafeRelease(m_cbuf_fusionConst);
	SafeRelease(m_cbuf_fusionPerFrame);
	SafeRelease(m_cbuf_marchingCubesConst);
	SafeRelease(m_cbuf_marchingCubesPerFrame);

	SafeRelease(m_shader_marchingCubesAttachNan);
	SafeRelease(m_shader_marchingCubes);
	SafeRelease(m_shader_fusion);

	SafeRelease(m_blob_marchingCubesAttachNan);
	SafeRelease(m_blob_fusionShader);
	SafeRelease(m_blob_marchingCubesShader);
	SafeRelease(m_swapChain)
	SafeRelease(m_d3dContext);
	SafeRelease(m_d3dDevice);
	#ifdef _DEBUG
	m_d3dDebug->ReportLiveDeviceObjects(D3D11_RLDO_DETAIL);
	#endif //_DEBUG
	SafeRelease(m_d3dDebug);
	DestroyWindow(m_hWindow);
	UnregisterClass("MarkerlessAR", m_hInstance);

	SAFE_DELETE(m_volume);
}

void FusionGPU::initialize(){
	m_volume = new Volume(Size(-4, -4, -4), Size(4, 4, 4), m_system_parameters.m_volume_size, 1, false);
	m_trunaction = m_volume->m_voxel_size * m_system_parameters.m_truncation_scaling;

	m_fusionSettings.m_max = m_volume->m_max.cast<float>();
	m_fusionSettings.m_min = m_volume->m_min.cast<float>();
	m_fusionSettings.m_resolution = m_volume->m_size;
	m_fusionSettings.m_resSQ = m_volume->m_size * m_volume->m_size;

	m_fusionSettings.m_truncation = m_trunaction;
	m_fusionSettings.m_voxel_size = m_volume->m_voxel_size;

	m_fusionSettings.m_image_height = m_system_parameters.m_image_height;
	m_fusionSettings.m_image_width = m_system_parameters.m_image_width;
	m_fusionSettings.m_cX = m_system_parameters.m_cX;
	m_fusionSettings.m_cY = m_system_parameters.m_cY;
	m_fusionSettings.m_focal_length_X = m_system_parameters.m_focal_length_X;
	m_fusionSettings.m_focal_length_Y = m_system_parameters.m_focal_length_Y;
}

void FusionGPU::integrate(std::shared_ptr<PointCloud> cloud){
	const auto cameraToWorld = cloud->m_pose_estimation;
	const auto worldToCamera = cameraToWorld.inverse();
	const auto frustum_box = computeFrustumBounds(cameraToWorld, cloud->m_system_parameters);

	m_fusionPerFrame.cam2world = cameraToWorld.transpose(); //col major -> row major
	m_fusionPerFrame.world2cam = worldToCamera.transpose(); //col major -> row major
	m_fusionPerFrame.frustum_max = frustum_box.m_max;
	m_fusionPerFrame.frustum_min = frustum_box.m_min;

	m_fusionPerFrame.numThreads = frustum_box.m_max - frustum_box.m_min;

	m_d3dContext->UpdateSubresource(m_cbuf_fusionPerFrame, 0, nullptr, &m_fusionPerFrame, 0, -1);

	m_d3dContext->UpdateSubresource(m_t2d_currentFrame, 0, nullptr, cloud->m_depth_points_fusion,
	                                sizeof(float) * m_fusionSettings.m_image_width, -1);

	ID3D11Buffer* buffers[] =
	{
		m_cbuf_fusionConst,
		m_cbuf_fusionPerFrame,
		nullptr, nullptr
	};
	ID3D11UnorderedAccessView* UAV[] = {m_uav_sdf, nullptr, nullptr};

	UINT initial[] = {0, 0, 0};

	m_d3dContext->CSSetConstantBuffers(0, 4, buffers);
	m_d3dContext->CSSetUnorderedAccessViews(0, 3, UAV, initial);
	m_d3dContext->CSSetShader(m_shader_fusion, nullptr, 0);
	m_d3dContext->Dispatch(
		(m_fusionPerFrame.numThreads.x() / FUSION_THREADS + 1),
		(m_fusionPerFrame.numThreads.y() / FUSION_THREADS + 1),
		(m_fusionPerFrame.numThreads.z()) / FUSION_THREADS + 1);

	//m_swapChain->Present(0, 0); //Debug hook 

}

void FusionGPU::save(std::string name){
	Mesh mesh;
	processMesh(mesh);
	mesh.save(name);

}

void FusionGPU::processMesh(Mesh& mesh){
	#ifdef USE_CPU_MC
	processMeshCPU(mesh); 
	#else

	ID3D11Buffer* m_buf_vertexBuffer_copy = nullptr;

	D3D11_BUFFER_DESC descCopyBuf = {0};
	descCopyBuf.ByteWidth = sizeof(__Triangle) * MAX_TRIANGLES;
	descCopyBuf.StructureByteStride = sizeof(__Triangle);
	descCopyBuf.CPUAccessFlags = 0;
	descCopyBuf.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
	descCopyBuf.Usage = D3D11_USAGE_STAGING;
	descCopyBuf.BindFlags = 0;
	descCopyBuf.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

	HRESULT hr = m_d3dDevice->CreateBuffer(&descCopyBuf, nullptr, &m_buf_vertexBuffer_copy);
	if (FAILED(hr))
	{
		std::cout << "failed to create SDF copy buffer" << std::endl;
	}

	D3D11_UNORDERED_ACCESS_VIEW_DESC descUAV;
	descUAV.Buffer.FirstElement = 0;
	descUAV.Buffer.NumElements = MAX_TRIANGLES;
	descUAV.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_APPEND;
	descUAV.Format = DXGI_FORMAT_UNKNOWN;
	descUAV.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;

	ID3D11UnorderedAccessView* uav_triangleBuffer = nullptr;

	hr = m_d3dDevice->CreateUnorderedAccessView(m_buf_vertexBuffer, &descUAV, &uav_triangleBuffer);
	if (FAILED(hr))
	{
		std::cout << "failed to UAV for MQ" << std::endl;
		std::cin.get();
	}

	ID3D11UnorderedAccessView* UAV[] = {m_uav_sdf, uav_triangleBuffer};
	UINT initial[] = {0, 0};
	ID3D11Buffer* buffers[] =
	{
		m_cbuf_fusionConst,
		m_cbuf_fusionPerFrame,
		m_cbuf_marchingCubesConst, nullptr
	};

	m_d3dContext->CSSetConstantBuffers(0, 4, buffers);
	m_d3dContext->CSSetUnorderedAccessViews(0, 2, UAV, initial);
	m_d3dContext->CSSetShader(m_shader_marchingCubes, nullptr, 0);

	m_d3dContext->Dispatch(
		(m_fusionSettings.m_resolution / MC_THREADS + 1),
		(m_fusionSettings.m_resolution / MC_THREADS + 1),
		(m_fusionSettings.m_resolution / MC_THREADS + 1));

	m_d3dContext->CSSetShader(m_shader_marchingCubesAttachNan, nullptr, 0);
	m_d3dContext->Dispatch(1, 1, 1);

	m_d3dContext->CopyResource(m_buf_vertexBuffer_copy, m_buf_vertexBuffer);

	D3D11_MAPPED_SUBRESOURCE verticesMap;
	hr = m_d3dContext->Map(m_buf_vertexBuffer_copy, 0, D3D11_MAP_READ, 0, &verticesMap);

	if (FAILED(hr))
	{
		std::cout << "failed to map SDF to system memory" << std::endl;
		std::cin.get();
	}

	mesh.m_vertices.clear();
	mesh.m_triangles.clear();

	__Triangle* pData = static_cast<__Triangle*>(verticesMap.pData);

	for (size_t i = 0; i < MAX_TRIANGLES; ++i)
	{
		__Triangle& t = *(pData + i);

		if (!t.v1.allFinite() || !t.v1.allFinite() || !t.v1.allFinite()) // dummy reached
		{
			break;
		}

		mesh.m_triangles.push_back(Triangle(3 * i, 3 * i + 1, 3 * i + 2)); // best index buffer ever!
		mesh.m_vertices.push_back(t.v0);
		mesh.m_vertices.push_back(t.v1);
		mesh.m_vertices.push_back(t.v2);
	}
	m_d3dContext->Unmap(m_buf_vertexBuffer_copy, 0);

	//	m_swapChain->Present(0, 0); //Debug hook

	SafeRelease(m_buf_vertexBuffer_copy);

	#endif //#ifdef USE_CPU_MC
}

void FusionGPU::processMeshCPU(Mesh& mesh){
	ID3D11Buffer* m_buf_sdf_copy = nullptr;
	D3D11_BUFFER_DESC descSDF = {0};
	descSDF.ByteWidth = sizeof(Voxel) * m_fusionSettings.m_resolution * m_fusionSettings.m_resolution * m_fusionSettings
		.m_resolution;
	descSDF.StructureByteStride = sizeof(Voxel);
	descSDF.CPUAccessFlags = 0;
	descSDF.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
	descSDF.Usage = D3D11_USAGE_STAGING;
	descSDF.BindFlags = 0;
	descSDF.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

	HRESULT hr = m_d3dDevice->CreateBuffer(&descSDF, nullptr, &m_buf_sdf_copy);
	if (FAILED(hr))
	{
		std::cout << "failed to create SDF copy buffer" << std::endl;
	}

	ID3D11UnorderedAccessView* nullUAV[] = {nullptr};
	m_d3dContext->CSSetUnorderedAccessViews(0, 1, nullUAV, nullptr);
	m_d3dContext->CSSetShader(nullptr, nullptr, 0);

	m_d3dContext->CopyResource(m_buf_sdf_copy, m_buf_sdf);

	D3D11_MAPPED_SUBRESOURCE sdfMap;
	hr = m_d3dContext->Map(m_buf_sdf_copy, 0, D3D11_MAP_READ, 0, &sdfMap);

	if (FAILED(hr))
	{
		std::cout << "failed to map SDF to system memory" << std::endl;
		std::cin.get();
	}

	m_volume->m_voxels = reinterpret_cast<Voxel*>(sdfMap.pData);

	#pragma omp parallel for num_threads(2)
	for (int x = 0; x < m_volume->m_size - 1; x++)
		for (int y = 0; y < m_volume->m_size - 1; y++)
			for (int z = 0; z < m_volume->m_size - 1; z++)
				MarchingCubes::getInstance().ProcessVolumeCell(m_volume, x, y, z, 0.0f, &mesh);

	m_d3dContext->Unmap(m_buf_sdf_copy, 0);

	SafeRelease(m_buf_sdf_copy);

}

void FusionGPU::populateSettingsBuffers(){
	m_d3dContext->UpdateSubresource(m_cbuf_fusionConst, 0, nullptr, &m_fusionSettings, 0, -1);
	m_d3dContext->UpdateSubresource(m_cbuf_marchingCubesConst, 0, nullptr, &m_marchingCubesSettings, 0, -1);
}

/****
*
*	************************************************ D3D INIT *****************************************
*
*/

void FusionGPU::initBuffers(){
	ID3DBlob* errBlob;
	HRESULT hr;

	D3D11_BUFFER_DESC descSDF = {0};
	descSDF.ByteWidth = sizeof(Voxel) * m_fusionSettings.m_resolution * m_fusionSettings.m_resolution * m_fusionSettings
		.m_resolution;
	descSDF.StructureByteStride = sizeof(Voxel);
	descSDF.Usage = D3D11_USAGE_DEFAULT;
	descSDF.CPUAccessFlags = 0;
	descSDF.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
	descSDF.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;

	unsigned char* zeroSDF = new unsigned char[sizeof(Voxel) * m_fusionSettings.m_resolution * m_fusionSettings.
		m_resolution * m_fusionSettings.m_resolution];
	ZeroMemory(zeroSDF, sizeof(Voxel) * m_fusionSettings.m_resolution * m_fusionSettings.m_resolution *m_fusionSettings.
m_resolution);
	D3D11_SUBRESOURCE_DATA dataZeroSdf;
	dataZeroSdf.pSysMem = zeroSDF;

	hr = m_d3dDevice->CreateBuffer(&descSDF, &dataZeroSdf, &m_buf_sdf);
	if (FAILED(hr))
	{
		std::cout << "failed to create buffer" << std::endl;
		std::cin.get();
	}
	delete[] zeroSDF;

	descSDF.ByteWidth = sizeof(__Triangle) * MAX_TRIANGLES;
	descSDF.StructureByteStride = sizeof(__Triangle);

	hr = m_d3dDevice->CreateBuffer(&descSDF, nullptr, &m_buf_vertexBuffer);
	if (FAILED(hr))
	{
		std::cout << "failed to create buffer" << std::endl;
		std::cin.get();
	}

	D3D11_UNORDERED_ACCESS_VIEW_DESC descSDFUAV;
	descSDFUAV.Buffer.FirstElement = 0;
	descSDFUAV.Buffer.Flags = 0;
	descSDFUAV.Buffer.NumElements = m_fusionSettings.m_resolution * m_fusionSettings.m_resolution * m_fusionSettings.
		m_resolution;
	descSDFUAV.Format = DXGI_FORMAT_UNKNOWN;
	descSDFUAV.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;

	hr = m_d3dDevice->CreateUnorderedAccessView(m_buf_sdf, &descSDFUAV, &m_uav_sdf);
	if (FAILED(hr))
	{
		std::cout << "failed to UAV" << std::endl;
		std::cin.get();
	}

	D3D11_TEXTURE2D_DESC descCurrentFrameBuffer = {0};
	descCurrentFrameBuffer.Width = m_system_parameters.m_image_width;
	descCurrentFrameBuffer.Height = m_system_parameters.m_image_height;
	descCurrentFrameBuffer.MipLevels = 0;
	descCurrentFrameBuffer.ArraySize = 1;
	descCurrentFrameBuffer.Format = DXGI_FORMAT_R32_FLOAT;
	descCurrentFrameBuffer.SampleDesc.Count = 1;
	descCurrentFrameBuffer.SampleDesc.Quality = 0;
	descCurrentFrameBuffer.Usage = D3D11_USAGE_DEFAULT;
	descCurrentFrameBuffer.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	descCurrentFrameBuffer.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr = m_d3dDevice->CreateTexture2D(&descCurrentFrameBuffer, nullptr, &m_t2d_currentFrame);
	if (FAILED(hr))
	{
		std::cout << "failed to create buffer" << std::endl;
		std::cin.get();
	}

	D3D11_SHADER_RESOURCE_VIEW_DESC descCurrentFrameSRV;
	descCurrentFrameSRV.Texture2D.MipLevels = 1;
	descCurrentFrameSRV.Texture2D.MostDetailedMip = 0;
	descCurrentFrameSRV.Format = DXGI_FORMAT_R32_FLOAT;
	descCurrentFrameSRV.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;

	hr = m_d3dDevice->CreateShaderResourceView(m_t2d_currentFrame, &descCurrentFrameSRV, &m_srv_currentFrame);
	if (FAILED(hr))
	{
		std::cout << "failed to create SRV" << std::endl;
		std::cin.get();
	}

	D3D11_BUFFER_DESC descBuffer = {0};
	descBuffer.Usage = D3D11_USAGE_DEFAULT;
	descBuffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	descBuffer.ByteWidth = sizeof(FusionSettings);

	hr = m_d3dDevice->CreateBuffer(&descBuffer, nullptr, &m_cbuf_fusionConst);
	if (FAILED(hr))
	{
		std::cout << "failed to create buffer" << std::endl;
		std::cin.get();
	}
	descBuffer.ByteWidth = sizeof(FusionPerFrame);

	hr = m_d3dDevice->CreateBuffer(&descBuffer, nullptr, &m_cbuf_fusionPerFrame);
	if (FAILED(hr))
	{
		std::cout << "failed to create buffer" << std::endl;
		std::cin.get();
	}

	descBuffer.ByteWidth = sizeof(MarchingCubesSettings);

	hr = m_d3dDevice->CreateBuffer(&descBuffer, nullptr, &m_cbuf_marchingCubesConst);
	if (FAILED(hr))
	{
		std::cout << "failed to create buffer" << std::endl;
		std::cin.get();
	}

	descBuffer.ByteWidth = sizeof(MarchingCubesPerFrame);

	hr = m_d3dDevice->CreateBuffer(&descBuffer, nullptr, &m_cbuf_marchingCubesPerFrame);
	if (FAILED(hr))
	{
		std::cout << "failed to create buffer" << std::endl;
		std::cin.get();
	}
}

void FusionGPU::reloadShaders(std::string shaderPath){
	ID3DBlob* errBlob;
	HRESULT hr;
	SafeRelease(m_shader_marchingCubesAttachNan);
	SafeRelease(m_shader_marchingCubes);
	SafeRelease(m_shader_fusion);

	SafeRelease(m_blob_marchingCubesAttachNan);
	SafeRelease(m_blob_fusionShader);
	SafeRelease(m_blob_marchingCubesShader);

	char current[FILENAME_MAX];
	_getcwd(current, sizeof(current));

	std::wstring wsShaderPath(shaderPath.begin(), shaderPath.end());

	hr = D3DCompileFromFile(wsShaderPath.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "CS_FUSION", "cs_5_0",
	                        D3DCOMPILE_DEBUG | D3DCOMPILE_ENABLE_STRICTNESS, NULL, &m_blob_fusionShader, &errBlob);
	if (FAILED(hr))
	{
		std::cout << "failed to compile Fusion shader " << std::endl;
		std::cout << (char*)errBlob->GetBufferPointer() << std::endl;
		std::cin.get();
	}

	hr = D3DCompileFromFile(wsShaderPath.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "CS_MC", "cs_5_0",
	                        D3DCOMPILE_DEBUG | D3DCOMPILE_ENABLE_STRICTNESS, NULL, &m_blob_marchingCubesShader,
	                        &errBlob);
	if (FAILED(hr))
	{
		std::cout << "failed to compile Marching Cubes shader " << std::endl;
		std::cout << (char*)errBlob->GetBufferPointer() << std::endl;
		std::cin.get();
	}

	hr = D3DCompileFromFile(wsShaderPath.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "CS_ATTACH_DUMMY",
	                        "cs_5_0", D3DCOMPILE_DEBUG | D3DCOMPILE_ENABLE_STRICTNESS, NULL,
	                        &m_blob_marchingCubesAttachNan, &errBlob);
	if (FAILED(hr))
	{
		std::cout << "failed to compile Marching Cubes DUMMY shader " << std::endl;
		std::cout << (char*)errBlob->GetBufferPointer() << std::endl;
		std::cin.get();
	}

	hr = m_d3dDevice->CreateComputeShader(m_blob_fusionShader->GetBufferPointer(), m_blob_fusionShader->GetBufferSize(),
	                                      nullptr, &m_shader_fusion);
	if (FAILED(hr))
	{
		std::cout << "failed to load Fusion Shader " << hr << std::endl;
		std::cin.get();
	}
	hr = m_d3dDevice->CreateComputeShader(m_blob_marchingCubesShader->GetBufferPointer(),
	                                      m_blob_marchingCubesShader->GetBufferSize(), nullptr,
	                                      &m_shader_marchingCubes);
	if (FAILED(hr))
	{
		std::cout << "failed to load Marching Cubes Shader " << hr << std::endl;
		std::cin.get();
	}

	hr = m_d3dDevice->CreateComputeShader(m_blob_marchingCubesAttachNan->GetBufferPointer(),
	                                      m_blob_marchingCubesAttachNan->GetBufferSize(), nullptr,
	                                      &m_shader_marchingCubesAttachNan);
	if (FAILED(hr))
	{
		std::cout << "failed to load Marching Cubes Shader " << hr << std::endl;
		std::cin.get();
	}

	m_d3dContext->CSSetShaderResources(0, 1, &m_srv_currentFrame);

}

void FusionGPU::initDx11(){
	HRESULT result;

	//Multisample AA
	DXGI_SAMPLE_DESC descSampling;
	descSampling.Count = 8;
	descSampling.Quality = 0;

	DXGI_SWAP_CHAIN_DESC descSwapChain;
	ZeroMemory(&descSwapChain, sizeof(DXGI_SWAP_CHAIN_DESC));
	descSwapChain.BufferDesc.Height = m_system_parameters.m_image_height;
	descSwapChain.BufferDesc.Width = m_system_parameters.m_image_width;
	descSwapChain.BufferDesc.RefreshRate.Denominator = 1;
	descSwapChain.BufferDesc.RefreshRate.Numerator = 60;
	descSwapChain.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	descSwapChain.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	descSwapChain.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	descSwapChain.BufferCount = 1;
	descSwapChain.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	descSwapChain.OutputWindow = m_hWindow;
	descSwapChain.SampleDesc = descSampling;
	descSwapChain.Windowed = true; //window / full screen
	descSwapChain.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

	#ifdef _DEBUG
	result = D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, D3D11_CREATE_DEVICE_DEBUG, NULL, NULL, D3D11_SDK_VERSION, &descSwapChain, &m_swapChain, &m_d3dDevice, NULL, &m_d3dContext);
	m_d3dDevice->QueryInterface(__uuidof(ID3D11Debug), (void**)&m_d3dDebug);

	#else
	result = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, NULL,
	                                       D3D11_SDK_VERSION, &descSwapChain, &m_swapChain, &m_d3dDevice, nullptr,
	                                       &m_d3dContext);
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

	if (!options.ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x)
	{
		MessageBox(nullptr, "Compute Shaders are not supported on your hardware", "Unsupported HW", MB_OK);
	}

}

//Callback function in case of window events
LRESULT CALLBACK windowProc(HWND hWindow, UINT message, WPARAM wParam, LPARAM lParam){

	if (message == WM_DESTROY)
	{
		// close the application entirely
		PostQuitMessage(0);
		return 0;
	}

	return DefWindowProc(hWindow, message, wParam, lParam);
}

void FusionGPU::initWindow(){
	//defining window class parameters
	WNDCLASSEX windowInfo;
	ZeroMemory(&windowInfo, sizeof(WNDCLASSEX));

	m_hInstance = GetModuleHandle(nullptr);

	windowInfo.cbSize = sizeof(WNDCLASSEX);
	windowInfo.style = CS_HREDRAW | CS_VREDRAW;
	windowInfo.lpfnWndProc = &windowProc;
	windowInfo.hInstance = m_hInstance;
	windowInfo.hbrBackground = (HBRUSH)COLOR_WINDOW; //remove for fullscreen rendering
	windowInfo.lpszClassName = "MarkerlessAR";
	windowInfo.hCursor = LoadCursor(nullptr, IDC_ARROW);
	windowInfo.hIcon = LoadIcon(nullptr, IDI_WINLOGO);
	windowInfo.hIconSm = LoadIcon(nullptr, IDI_WINLOGO);

	//registering window class
	if (!RegisterClassEx(&windowInfo))
	{
		MessageBox(nullptr, "Failed to register Window class! ", "ERROR!", MB_OK);
		std::exit(101);
	}

	//creating the window the game is going to be rendered to

	m_hWindow = CreateWindowEx(
		NULL,
		windowInfo.lpszClassName,
		"MarkerlessAR",
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT,
		m_system_parameters.m_image_width, m_system_parameters.m_image_height,
		nullptr,
		nullptr,
		m_hInstance,
		nullptr);

	if (!m_hWindow)
	{
		MessageBox(nullptr, "Failed to create Window! ", "ERROR!", MB_OK);
		std::exit(102);
	}

	ShowWindow(m_hWindow, false);

	UpdateWindow(m_hWindow);
}

#pragma pack_matrix(row_major)


/*
* ****************************** STRUCTS **************************
*/
struct FusionSettings
{    
    float3 m_min;
    float3 m_max;
    float m_truncation;
    float m_voxel_size;
    float m_focal_length_X;
    float m_focal_length_Y;
    float m_cX;
    float m_cY;
    float m_depth_min;
    float m_depth_max;

    int m_image_height;
    int m_image_width;
    int m_resolution;
};

struct FusionPerFrame
{
    float4x4 cam2world;
    float4x4 world2cam;
    int3 frustum_min;
    int3 frustum_max;
    int3 num_threads;
};

struct Voxel
{
    float sdf;
    float weight;
    int state;
};

/*
* ****************************** BUFFERS **************************
*/

cbuffer constSettings : register(b0)
{
    FusionSettings g_settings;
};

cbuffer perFrameSettings : register(b1)
{
    FusionPerFrame g_perFrame;
};

RWStructuredBuffer<Voxel> g_SDF; 

Texture2D<float> g_currentFrame : register(t1);




void main(uint3 threadIDInGroup : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    int3 cellIDX3 = g_perFrame.frustum_min + threadIDInGroup; 

    int cellIDX = dot(cellIDX3, int3(1, 1, 1)); 
    
    

}
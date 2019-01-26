#pragma pack_matrix(row_major)

#define THREADS_PER_GROUP_DIM 4 

#define VOXEL_UNSEEN 0
#define VOXEL_EMPTY 1
#define VOXEL_SDF 2

#define MAX_DEPTH 5.0f
#define MAX_WEIGHT 1000.0f

/*
* ****************************** STRUCTS **************************
*/
struct FusionSettings
{    
    float3 m_min;
    float3 m_max;
    float m_truncation;
    float m_voxel_size;
    float2 m_focal_length;
    float2 m_principalpt;
    float m_depth_min;
    float m_depth_max;

    int2 imageDims; 
    int m_resolution;
    int m_resSQ; 
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

RWStructuredBuffer<Voxel> g_SDF : register(u0);

Texture2D<float> g_currentFrame : register(t1);

/*
* ****************************** Functions **************************
*/

float weightKernel(float depth, float max)
{
    if (depth <= 0.01f)
        return 1.f;
    return 1.f - depth / max;
}



/*
* ****************************** Shaders **************************
*/

[numthreads(THREADS_PER_GROUP_DIM, THREADS_PER_GROUP_DIM, THREADS_PER_GROUP_DIM)]
void main(uint3 threadIDInGroup : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    int3 cellIDX3 = g_perFrame.frustum_min + groupID * int3(THREADS_PER_GROUP_DIM, THREADS_PER_GROUP_DIM, THREADS_PER_GROUP_DIM) + threadIDInGroup;
    float4 worldPos; 

    worldPos.xyz = g_settings.m_min + (g_settings.m_max - g_settings.m_min) * (float3) cellIDX3 / float3(g_settings.m_resolution, g_settings.m_resolution, g_settings.m_resolution);
    worldPos.w = 1; 

    float3 cell = mul(g_perFrame.world2cam, worldPos).xyz; 

    cell.xy = cell.xy * g_settings.m_focal_length / cell.zz + g_settings.m_principalpt; 
       
    int2 pixels = (int2) round(cell.xy); 

  //  if (all(pixels > int2(0, 0) && pixels < g_settings.imageDims))
    {
        float depth = g_currentFrame.Load(int3(pixels.xy, 0)); 


        if (depth < MAX_DEPTH)
        {
            int cellIDX = dot(cellIDX3, int3(g_settings.m_resSQ, g_settings.m_resolution, 1));

            float sdf = depth - cell.z; 

            if(depth - g_settings.m_truncation > cell.z)
            {
                g_SDF[cellIDX].state = VOXEL_EMPTY; 
            }
            else if (abs(sdf) < g_settings.m_truncation)
            {
                float weight = weightKernel(depth, g_settings.m_depth_max); 
                Voxel v = g_SDF[cellIDX];
                v.state = VOXEL_SDF; 
                v.sdf = v.sdf * v.weight + sdf * weight / (v.weight + weight);
                v.weight = min(v.weight + weight, MAX_WEIGHT); 

                g_SDF[cellIDX] = v; 
            }     

        } // depth < INFINITY
    }//pixel in image

}
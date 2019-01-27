#pragma pack_matrix(row_major)

#define THREADS_PER_GROUP_DIM 1 

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
    float m_truncation;
    float3 m_max;
    float m_voxel_size;
    float2 m_focal_length;
    float2 m_principalpt;

    int2 imageDims; 
    int m_resSQ;
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

cbuffer MQSettings : register(b3)
{
    int dummy; 
};

cbuffer MQPerFrame : register(b4)
{
    int dummy2; 
};

RWStructuredBuffer<Voxel> g_SDF : register(u0);

Texture2D<float> g_currentFrame : register(t0);

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

[numthreads(1, 1, 1)]
void main(uint3 threadIDInGroup : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    int3 cellIDX3 = g_perFrame.frustum_min + groupID;
    int cellIDX = dot(cellIDX3, int3(g_settings.m_resSQ, g_settings.m_resolution, 1));

    //g_SDF[cellIDX].sdf++;
    ////if (!any(groupID))
    ////{
    ////    g_SDF[0].sdf = g_perFrame.frustum_min.x;
    ////    g_SDF[0].weight = g_perFrame.frustum_min.y;
    ////    g_SDF[0].state = g_perFrame.frustum_min.z;
    ////    g_SDF[1].sdf = g_perFrame.frustum_max.x;
    ////    g_SDF[1].weight = g_perFrame.frustum_max.y;
    ////    g_SDF[1].state = g_perFrame.frustum_max.z;
    ////    g_SDF[2].sdf = cellIDX3.x;
    ////    g_SDF[2].weight = cellIDX3.y;
    ////    g_SDF[2].state = cellIDX3.z;
    ////    g_SDF[3].sdf = g_settings.m_resSQ;
    ////    g_SDF[3].weight = g_settings.m_resolution;
    ////    g_SDF[3].state = 1;
    ////    g_SDF[4].state = cellIDX;

    ////}
    

    //if (cellIDX >= g_settings.m_resSQ * g_settings.m_resolution)
    //{
    //    g_SDF[0].state++;

    //}
    //return; 
    //return;

    float4 cell; 

    float invScaling = g_settings.m_resolution - 1;
    cell.xyz = g_settings.m_min + (g_settings.m_max - g_settings.m_min) * (float3(cellIDX3) / invScaling);
    cell.w = 1; 

    cell = mul(g_perFrame.world2cam, cell); 

    cell.xy = cell.xy * g_settings.m_focal_length / cell.zz + g_settings.m_principalpt; 
       
    int2 pixels = (int2) round(cell.xy); 

    if (all(pixels >= int2(0, 0) && pixels < g_settings.imageDims))
    {
        float depth = g_currentFrame.Load(int3(pixels.xy, 0)); 

        if (depth < MAX_DEPTH)
        {
            float sdf = depth - cell.z; 


            if (depth - g_settings.m_truncation > cell.z)
            {
                g_SDF[cellIDX].state = VOXEL_EMPTY;
            }
            else if (abs(sdf) < g_settings.m_truncation)
            {
                float weight = weightKernel(depth, 5.0f); 
                Voxel v = g_SDF[cellIDX];
                v.state = VOXEL_SDF; 
                v.sdf = v.sdf * v.weight + sdf * weight / (v.weight + weight);
                v.weight = min(v.weight + weight, MAX_WEIGHT); 

                g_SDF[cellIDX] = v; 
            }     

        } // depth < INFINITY
    }//pixel in image

}
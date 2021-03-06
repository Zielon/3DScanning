#if 0
//
// Generated by Microsoft (R) HLSL Shader Compiler 10.1
//
//
// Buffer Definitions: 
//
// cbuffer constSettings
// {
//
//   struct FusionSettings
//   {
//       
//       float3 m_min;                  // Offset:    0
//       float m_truncation;            // Offset:   12
//       float3 m_max;                  // Offset:   16
//       float m_voxel_size;            // Offset:   28
//       float2 m_focal_length;         // Offset:   32
//       float2 m_principalpt;          // Offset:   40
//       int2 imageDims;                // Offset:   48
//       float m_max_depth;             // Offset:   56
//       int m_resolution;              // Offset:   60
//
//   } g_settings;                      // Offset:    0 Size:    64
//
// }
//
// cbuffer perFrameSettings
// {
//
//   struct FusionPerFrame
//   {
//       
//       row_major float4x4 cam2world;  // Offset:    0
//       row_major float4x4 world2cam;  // Offset:   64
//       int3 frustum_min;              // Offset:  128
//       int3 frustum_max;              // Offset:  144
//       int3 num_threads;              // Offset:  160
//
//   } g_perFrame;                      // Offset:    0 Size:   172
//
// }
//
// Resource bind info for g_SDF
// {
//
//   struct Voxel
//   {
//       
//       float sdf;                     // Offset:    0
//       float weight;                  // Offset:    4
//       int state;                     // Offset:    8
//
//   } $Element;                        // Offset:    0 Size:    12
//
// }
//
//
// Resource Bindings:
//
// Name                                 Type  Format         Dim      HLSL Bind  Count
// ------------------------------ ---------- ------- ----------- -------------- ------
// g_currentFrame                    texture   float          2d             t0      1 
// g_SDF                                 UAV  struct         r/w             u0      1 
// constSettings                     cbuffer      NA          NA            cb0      1 
// perFrameSettings                  cbuffer      NA          NA            cb1      1 
//
//
//
// Input signature:
//
// Name                 Index   Mask Register SysValue  Format   Used
// -------------------- ----- ------ -------- -------- ------- ------
// no Input
//
// Output signature:
//
// Name                 Index   Mask Register SysValue  Format   Used
// -------------------- ----- ------ -------- -------- ------- ------
// no Output
cs_5_0
dcl_globalFlags refactoringAllowed
dcl_constantbuffer CB0[4], immediateIndexed
dcl_constantbuffer CB1[9], immediateIndexed
dcl_resource_texture2d (float,float,float,float) t0
dcl_uav_structured u0, 12
dcl_input vThreadGroupID.xyz
dcl_input vThreadIDInGroup.xyz
dcl_temps 3
dcl_thread_group 4, 4, 4
ishl r0.xyz, vThreadGroupID.xyzx, l(2, 2, 2, 0)
iadd r0.xyz, r0.xyzx, cb1[8].xyzx
iadd r0.xyz, r0.xyzx, vThreadIDInGroup.xyzx
iadd r0.w, cb0[3].w, l(-1)
itof r0.w, r0.w
add r1.xyz, -cb0[0].xyzx, cb0[1].xyzx
itof r2.xyz, r0.xyzx
div r2.xyz, r2.xyzx, r0.wwww
mad r1.xyz, r1.xyzx, r2.xyzx, cb0[0].xyzx
mov r1.w, l(1.000000)
dp4 r2.x, cb1[4].xyzw, r1.xyzw
dp4 r2.y, cb1[5].xyzw, r1.xyzw
dp4 r0.w, cb1[6].xyzw, r1.xyzw
mul r1.xy, r2.xyxx, cb0[2].xyxx
div r1.xy, r1.xyxx, r0.wwww
add r1.xy, r1.xyxx, cb0[2].zwzz
round_ne r1.xy, r1.xyxx
ftoi r1.xy, r1.xyxx
ige r2.xy, r1.xyxx, l(0, 0, 0, 0)
ilt r2.zw, r1.xxxy, cb0[3].xxxy
and r2.xy, r2.zwzz, r2.xyxx
and r2.x, r2.y, r2.x
if_nz r2.x
  mov r1.zw, l(0,0,0,0)
  ld_indexable(texture2d)(float,float,float,float) r1.x, r1.xyzw, t0.xyzw
  lt r1.y, l(0.001000), r1.x
  lt r1.z, r1.x, cb0[3].z
  and r1.y, r1.z, r1.y
  if_nz r1.y
    imul null, r1.y, cb0[3].w, cb0[3].w
    imul null, r0.y, r0.y, cb0[3].w
    imad r0.x, r0.x, r1.y, r0.y
    iadd r0.x, r0.z, r0.x
    add r0.y, r1.x, -cb0[0].w
    lt r0.y, r0.w, r0.y
    ld_structured_indexable(structured_buffer, stride=12)(mixed,mixed,mixed,mixed) r0.z, r0.x, l(8), u0.xxxx
    ieq r1.y, r0.z, l(0)
    and r0.y, r0.y, r1.y
    if_nz r0.y
      store_structured u0.x, r0.x, l(8), l(1)
    else 
      add r0.y, -r0.w, r1.x
      lt r0.w, |r0.y|, cb0[0].w
      and r0.z, r1.y, r0.w
      if_nz r0.z
        lt r0.z, l(0.010000), r1.x
        div r0.w, r1.x, cb0[3].z
        add r0.w, -r0.w, l(1.000000)
        movc r0.z, r0.z, r0.w, l(1.000000)
        ld_structured_indexable(structured_buffer, stride=12)(mixed,mixed,mixed,mixed) r1.xy, r0.x, l(0), u0.xyxx
        mul r0.y, r0.z, r0.y
        add r0.z, r0.z, r1.y
        div r0.y, r0.y, r0.z
        mad r1.x, r1.x, r1.y, r0.y
        min r1.y, r0.z, l(50000.000000)
        mov r1.z, l(2)
        store_structured u0.xyz, r0.x, l(0), r1.xyzx
      endif 
    endif 
  endif 
endif 
ret 
// Approximately 62 instruction slots used
#endif

const BYTE g_CS_FUSION[] =
{
     68,  88,  66,  67,  35,  84, 
     72,   6, 234, 189,  70,  14, 
    126,   3, 219, 193, 151,  86, 
    250,  43,   1,   0,   0,   0, 
    144,  13,   0,   0,   5,   0, 
      0,   0,  52,   0,   0,   0, 
    192,   5,   0,   0, 208,   5, 
      0,   0, 224,   5,   0,   0, 
    244,  12,   0,   0,  82,  68, 
     69,  70, 132,   5,   0,   0, 
      3,   0,   0,   0, 240,   0, 
      0,   0,   4,   0,   0,   0, 
     60,   0,   0,   0,   0,   5, 
     83,  67,   0,   1,   0,   0, 
     92,   5,   0,   0,  82,  68, 
     49,  49,  60,   0,   0,   0, 
     24,   0,   0,   0,  32,   0, 
      0,   0,  40,   0,   0,   0, 
     36,   0,   0,   0,  12,   0, 
      0,   0,   0,   0,   0,   0, 
    188,   0,   0,   0,   2,   0, 
      0,   0,   5,   0,   0,   0, 
      4,   0,   0,   0, 255, 255, 
    255, 255,   0,   0,   0,   0, 
      1,   0,   0,   0,   1,   0, 
      0,   0, 203,   0,   0,   0, 
      6,   0,   0,   0,   6,   0, 
      0,   0,   1,   0,   0,   0, 
     12,   0,   0,   0,   0,   0, 
      0,   0,   1,   0,   0,   0, 
      1,   0,   0,   0, 209,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   1,   0, 
      0,   0,   1,   0,   0,   0, 
    223,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   1,   0,   0,   0, 
      1,   0,   0,   0,   1,   0, 
      0,   0, 103,  95,  99, 117, 
    114, 114, 101, 110, 116,  70, 
    114,  97, 109, 101,   0, 103, 
     95,  83,  68,  70,   0,  99, 
    111, 110, 115, 116,  83, 101, 
    116, 116, 105, 110, 103, 115, 
      0, 112, 101, 114,  70, 114, 
     97, 109, 101,  83, 101, 116, 
    116, 105, 110, 103, 115,   0, 
    209,   0,   0,   0,   1,   0, 
      0,   0,  56,   1,   0,   0, 
     64,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
    223,   0,   0,   0,   1,   0, 
      0,   0,  76,   3,   0,   0, 
    176,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
    203,   0,   0,   0,   1,   0, 
      0,   0, 128,   4,   0,   0, 
     12,   0,   0,   0,   0,   0, 
      0,   0,   3,   0,   0,   0, 
     96,   1,   0,   0,   0,   0, 
      0,   0,  64,   0,   0,   0, 
      2,   0,   0,   0,  40,   3, 
      0,   0,   0,   0,   0,   0, 
    255, 255, 255, 255,   0,   0, 
      0,   0, 255, 255, 255, 255, 
      0,   0,   0,   0, 103,  95, 
    115, 101, 116, 116, 105, 110, 
    103, 115,   0,  70, 117, 115, 
    105, 111, 110,  83, 101, 116, 
    116, 105, 110, 103, 115,   0, 
    109,  95, 109, 105, 110,   0, 
    102, 108, 111,  97, 116,  51, 
      0, 171,   1,   0,   3,   0, 
      1,   0,   3,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0, 128,   1, 
      0,   0, 109,  95, 116, 114, 
    117, 110,  99,  97, 116, 105, 
    111, 110,   0, 102, 108, 111, 
     97, 116,   0, 171,   0,   0, 
      3,   0,   1,   0,   1,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
    185,   1,   0,   0, 109,  95, 
    109,  97, 120,   0, 109,  95, 
    118, 111, 120, 101, 108,  95, 
    115, 105, 122, 101,   0, 109, 
     95, 102, 111,  99,  97, 108, 
     95, 108, 101, 110, 103, 116, 
    104,   0, 102, 108, 111,  97, 
    116,  50,   0, 171, 171, 171, 
      1,   0,   3,   0,   1,   0, 
      2,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   6,   2,   0,   0, 
    109,  95, 112, 114, 105, 110, 
     99, 105, 112,  97, 108, 112, 
    116,   0, 105, 109,  97, 103, 
    101,  68, 105, 109, 115,   0, 
    105, 110, 116,  50,   0, 171, 
    171, 171,   1,   0,   2,   0, 
      1,   0,   2,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,  76,   2, 
      0,   0, 109,  95, 109,  97, 
    120,  95, 100, 101, 112, 116, 
    104,   0, 109,  95, 114, 101, 
    115, 111, 108, 117, 116, 105, 
    111, 110,   0, 105, 110, 116, 
      0, 171, 171, 171,   0,   0, 
      2,   0,   1,   0,   1,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
    145,   2,   0,   0, 122,   1, 
      0,   0, 136,   1,   0,   0, 
      0,   0,   0,   0, 172,   1, 
      0,   0, 192,   1,   0,   0, 
     12,   0,   0,   0, 228,   1, 
      0,   0, 136,   1,   0,   0, 
     16,   0,   0,   0, 234,   1, 
      0,   0, 192,   1,   0,   0, 
     28,   0,   0,   0, 247,   1, 
      0,   0,  16,   2,   0,   0, 
     32,   0,   0,   0,  52,   2, 
      0,   0,  16,   2,   0,   0, 
     40,   0,   0,   0,  66,   2, 
      0,   0,  84,   2,   0,   0, 
     48,   0,   0,   0, 120,   2, 
      0,   0, 192,   1,   0,   0, 
     56,   0,   0,   0, 132,   2, 
      0,   0, 152,   2,   0,   0, 
     60,   0,   0,   0,   5,   0, 
      0,   0,   1,   0,  16,   0, 
      0,   0,   9,   0, 188,   2, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
    107,   1,   0,   0, 116,   3, 
      0,   0,   0,   0,   0,   0, 
    172,   0,   0,   0,   2,   0, 
      0,   0,  92,   4,   0,   0, 
      0,   0,   0,   0, 255, 255, 
    255, 255,   0,   0,   0,   0, 
    255, 255, 255, 255,   0,   0, 
      0,   0, 103,  95, 112, 101, 
    114,  70, 114,  97, 109, 101, 
      0,  70, 117, 115, 105, 111, 
    110,  80, 101, 114,  70, 114, 
     97, 109, 101,   0,  99,  97, 
    109,  50, 119, 111, 114, 108, 
    100,   0, 102, 108, 111,  97, 
    116,  52, 120,  52,   0, 171, 
    171, 171,   2,   0,   3,   0, 
      4,   0,   4,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0, 152,   3, 
      0,   0, 119, 111, 114, 108, 
    100,  50,  99,  97, 109,   0, 
    102, 114, 117, 115, 116, 117, 
    109,  95, 109, 105, 110,   0, 
    105, 110, 116,  51,   0, 171, 
      1,   0,   2,   0,   1,   0, 
      3,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0, 222,   3,   0,   0, 
    102, 114, 117, 115, 116, 117, 
    109,  95, 109,  97, 120,   0, 
    110, 117, 109,  95, 116, 104, 
    114, 101,  97, 100, 115,   0, 
    142,   3,   0,   0, 164,   3, 
      0,   0,   0,   0,   0,   0, 
    200,   3,   0,   0, 164,   3, 
      0,   0,  64,   0,   0,   0, 
    210,   3,   0,   0, 228,   3, 
      0,   0, 128,   0,   0,   0, 
      8,   4,   0,   0, 228,   3, 
      0,   0, 144,   0,   0,   0, 
     20,   4,   0,   0, 228,   3, 
      0,   0, 160,   0,   0,   0, 
      5,   0,   0,   0,   1,   0, 
     41,   0,   0,   0,   5,   0, 
     32,   4,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0, 127,   3,   0,   0, 
    168,   4,   0,   0,   0,   0, 
      0,   0,  12,   0,   0,   0, 
      2,   0,   0,   0,  56,   5, 
      0,   0,   0,   0,   0,   0, 
    255, 255, 255, 255,   0,   0, 
      0,   0, 255, 255, 255, 255, 
      0,   0,   0,   0,  36,  69, 
    108, 101, 109, 101, 110, 116, 
      0,  86, 111, 120, 101, 108, 
      0, 115, 100, 102,   0, 171, 
      0,   0,   3,   0,   1,   0, 
      1,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0, 185,   1,   0,   0, 
    119, 101, 105, 103, 104, 116, 
      0, 115, 116,  97, 116, 101, 
      0, 171, 171, 171,   0,   0, 
      2,   0,   1,   0,   1,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
    145,   2,   0,   0, 183,   4, 
      0,   0, 188,   4,   0,   0, 
      0,   0,   0,   0, 224,   4, 
      0,   0, 188,   4,   0,   0, 
      4,   0,   0,   0, 231,   4, 
      0,   0, 240,   4,   0,   0, 
      8,   0,   0,   0,   5,   0, 
      0,   0,   1,   0,   3,   0, 
      0,   0,   3,   0,  20,   5, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
    177,   4,   0,   0,  77, 105, 
     99, 114, 111, 115, 111, 102, 
    116,  32,  40,  82,  41,  32, 
     72,  76,  83,  76,  32,  83, 
    104,  97, 100, 101, 114,  32, 
     67, 111, 109, 112, 105, 108, 
    101, 114,  32,  49,  48,  46, 
     49,   0,  73,  83,  71,  78, 
      8,   0,   0,   0,   0,   0, 
      0,   0,   8,   0,   0,   0, 
     79,  83,  71,  78,   8,   0, 
      0,   0,   0,   0,   0,   0, 
      8,   0,   0,   0,  83,  72, 
     69,  88,  12,   7,   0,   0, 
     80,   0,   5,   0, 195,   1, 
      0,   0, 106,   8,   0,   1, 
     89,   0,   0,   4,  70, 142, 
     32,   0,   0,   0,   0,   0, 
      4,   0,   0,   0,  89,   0, 
      0,   4,  70, 142,  32,   0, 
      1,   0,   0,   0,   9,   0, 
      0,   0,  88,  24,   0,   4, 
      0, 112,  16,   0,   0,   0, 
      0,   0,  85,  85,   0,   0, 
    158,   0,   0,   4,   0, 224, 
     17,   0,   0,   0,   0,   0, 
     12,   0,   0,   0,  95,   0, 
      0,   2, 114,  16,   2,   0, 
     95,   0,   0,   2, 114,  32, 
      2,   0, 104,   0,   0,   2, 
      3,   0,   0,   0, 155,   0, 
      0,   4,   4,   0,   0,   0, 
      4,   0,   0,   0,   4,   0, 
      0,   0,  41,   0,   0,   9, 
    114,   0,  16,   0,   0,   0, 
      0,   0,  70,  18,   2,   0, 
      2,  64,   0,   0,   2,   0, 
      0,   0,   2,   0,   0,   0, 
      2,   0,   0,   0,   0,   0, 
      0,   0,  30,   0,   0,   8, 
    114,   0,  16,   0,   0,   0, 
      0,   0,  70,   2,  16,   0, 
      0,   0,   0,   0,  70, 130, 
     32,   0,   1,   0,   0,   0, 
      8,   0,   0,   0,  30,   0, 
      0,   6, 114,   0,  16,   0, 
      0,   0,   0,   0,  70,   2, 
     16,   0,   0,   0,   0,   0, 
     70,  34,   2,   0,  30,   0, 
      0,   8, 130,   0,  16,   0, 
      0,   0,   0,   0,  58, 128, 
     32,   0,   0,   0,   0,   0, 
      3,   0,   0,   0,   1,  64, 
      0,   0, 255, 255, 255, 255, 
     43,   0,   0,   5, 130,   0, 
     16,   0,   0,   0,   0,   0, 
     58,   0,  16,   0,   0,   0, 
      0,   0,   0,   0,   0,  10, 
    114,   0,  16,   0,   1,   0, 
      0,   0,  70, 130,  32, 128, 
     65,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
     70, 130,  32,   0,   0,   0, 
      0,   0,   1,   0,   0,   0, 
     43,   0,   0,   5, 114,   0, 
     16,   0,   2,   0,   0,   0, 
     70,   2,  16,   0,   0,   0, 
      0,   0,  14,   0,   0,   7, 
    114,   0,  16,   0,   2,   0, 
      0,   0,  70,   2,  16,   0, 
      2,   0,   0,   0, 246,  15, 
     16,   0,   0,   0,   0,   0, 
     50,   0,   0,  10, 114,   0, 
     16,   0,   1,   0,   0,   0, 
     70,   2,  16,   0,   1,   0, 
      0,   0,  70,   2,  16,   0, 
      2,   0,   0,   0,  70, 130, 
     32,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,  54,   0, 
      0,   5, 130,   0,  16,   0, 
      1,   0,   0,   0,   1,  64, 
      0,   0,   0,   0, 128,  63, 
     17,   0,   0,   8,  18,   0, 
     16,   0,   2,   0,   0,   0, 
     70, 142,  32,   0,   1,   0, 
      0,   0,   4,   0,   0,   0, 
     70,  14,  16,   0,   1,   0, 
      0,   0,  17,   0,   0,   8, 
     34,   0,  16,   0,   2,   0, 
      0,   0,  70, 142,  32,   0, 
      1,   0,   0,   0,   5,   0, 
      0,   0,  70,  14,  16,   0, 
      1,   0,   0,   0,  17,   0, 
      0,   8, 130,   0,  16,   0, 
      0,   0,   0,   0,  70, 142, 
     32,   0,   1,   0,   0,   0, 
      6,   0,   0,   0,  70,  14, 
     16,   0,   1,   0,   0,   0, 
     56,   0,   0,   8,  50,   0, 
     16,   0,   1,   0,   0,   0, 
     70,   0,  16,   0,   2,   0, 
      0,   0,  70, 128,  32,   0, 
      0,   0,   0,   0,   2,   0, 
      0,   0,  14,   0,   0,   7, 
     50,   0,  16,   0,   1,   0, 
      0,   0,  70,   0,  16,   0, 
      1,   0,   0,   0, 246,  15, 
     16,   0,   0,   0,   0,   0, 
      0,   0,   0,   8,  50,   0, 
     16,   0,   1,   0,   0,   0, 
     70,   0,  16,   0,   1,   0, 
      0,   0, 230, 138,  32,   0, 
      0,   0,   0,   0,   2,   0, 
      0,   0,  64,   0,   0,   5, 
     50,   0,  16,   0,   1,   0, 
      0,   0,  70,   0,  16,   0, 
      1,   0,   0,   0,  27,   0, 
      0,   5,  50,   0,  16,   0, 
      1,   0,   0,   0,  70,   0, 
     16,   0,   1,   0,   0,   0, 
     33,   0,   0,  10,  50,   0, 
     16,   0,   2,   0,   0,   0, 
     70,   0,  16,   0,   1,   0, 
      0,   0,   2,  64,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,  34,   0, 
      0,   8, 194,   0,  16,   0, 
      2,   0,   0,   0,   6,   4, 
     16,   0,   1,   0,   0,   0, 
      6, 132,  32,   0,   0,   0, 
      0,   0,   3,   0,   0,   0, 
      1,   0,   0,   7,  50,   0, 
     16,   0,   2,   0,   0,   0, 
    230,  10,  16,   0,   2,   0, 
      0,   0,  70,   0,  16,   0, 
      2,   0,   0,   0,   1,   0, 
      0,   7,  18,   0,  16,   0, 
      2,   0,   0,   0,  26,   0, 
     16,   0,   2,   0,   0,   0, 
     10,   0,  16,   0,   2,   0, 
      0,   0,  31,   0,   4,   3, 
     10,   0,  16,   0,   2,   0, 
      0,   0,  54,   0,   0,   8, 
    194,   0,  16,   0,   1,   0, 
      0,   0,   2,  64,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,  45,   0, 
      0, 137, 194,   0,   0, 128, 
     67,  85,  21,   0,  18,   0, 
     16,   0,   1,   0,   0,   0, 
     70,  14,  16,   0,   1,   0, 
      0,   0,  70, 126,  16,   0, 
      0,   0,   0,   0,  49,   0, 
      0,   7,  34,   0,  16,   0, 
      1,   0,   0,   0,   1,  64, 
      0,   0, 111,  18, 131,  58, 
     10,   0,  16,   0,   1,   0, 
      0,   0,  49,   0,   0,   8, 
     66,   0,  16,   0,   1,   0, 
      0,   0,  10,   0,  16,   0, 
      1,   0,   0,   0,  42, 128, 
     32,   0,   0,   0,   0,   0, 
      3,   0,   0,   0,   1,   0, 
      0,   7,  34,   0,  16,   0, 
      1,   0,   0,   0,  42,   0, 
     16,   0,   1,   0,   0,   0, 
     26,   0,  16,   0,   1,   0, 
      0,   0,  31,   0,   4,   3, 
     26,   0,  16,   0,   1,   0, 
      0,   0,  38,   0,   0,  10, 
      0, 208,   0,   0,  34,   0, 
     16,   0,   1,   0,   0,   0, 
     58, 128,  32,   0,   0,   0, 
      0,   0,   3,   0,   0,   0, 
     58, 128,  32,   0,   0,   0, 
      0,   0,   3,   0,   0,   0, 
     38,   0,   0,   9,   0, 208, 
      0,   0,  34,   0,  16,   0, 
      0,   0,   0,   0,  26,   0, 
     16,   0,   0,   0,   0,   0, 
     58, 128,  32,   0,   0,   0, 
      0,   0,   3,   0,   0,   0, 
     35,   0,   0,   9,  18,   0, 
     16,   0,   0,   0,   0,   0, 
     10,   0,  16,   0,   0,   0, 
      0,   0,  26,   0,  16,   0, 
      1,   0,   0,   0,  26,   0, 
     16,   0,   0,   0,   0,   0, 
     30,   0,   0,   7,  18,   0, 
     16,   0,   0,   0,   0,   0, 
     42,   0,  16,   0,   0,   0, 
      0,   0,  10,   0,  16,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   9,  34,   0,  16,   0, 
      0,   0,   0,   0,  10,   0, 
     16,   0,   1,   0,   0,   0, 
     58, 128,  32, 128,  65,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,  49,   0, 
      0,   7,  34,   0,  16,   0, 
      0,   0,   0,   0,  58,   0, 
     16,   0,   0,   0,   0,   0, 
     26,   0,  16,   0,   0,   0, 
      0,   0, 167,   0,   0, 139, 
      2,  99,   0, 128, 131, 153, 
     25,   0,  66,   0,  16,   0, 
      0,   0,   0,   0,  10,   0, 
     16,   0,   0,   0,   0,   0, 
      1,  64,   0,   0,   8,   0, 
      0,   0,   6, 224,  17,   0, 
      0,   0,   0,   0,  32,   0, 
      0,   7,  34,   0,  16,   0, 
      1,   0,   0,   0,  42,   0, 
     16,   0,   0,   0,   0,   0, 
      1,  64,   0,   0,   0,   0, 
      0,   0,   1,   0,   0,   7, 
     34,   0,  16,   0,   0,   0, 
      0,   0,  26,   0,  16,   0, 
      0,   0,   0,   0,  26,   0, 
     16,   0,   1,   0,   0,   0, 
     31,   0,   4,   3,  26,   0, 
     16,   0,   0,   0,   0,   0, 
    168,   0,   0,   9,  18, 224, 
     17,   0,   0,   0,   0,   0, 
     10,   0,  16,   0,   0,   0, 
      0,   0,   1,  64,   0,   0, 
      8,   0,   0,   0,   1,  64, 
      0,   0,   1,   0,   0,   0, 
     18,   0,   0,   1,   0,   0, 
      0,   8,  34,   0,  16,   0, 
      0,   0,   0,   0,  58,   0, 
     16, 128,  65,   0,   0,   0, 
      0,   0,   0,   0,  10,   0, 
     16,   0,   1,   0,   0,   0, 
     49,   0,   0,   9, 130,   0, 
     16,   0,   0,   0,   0,   0, 
     26,   0,  16, 128, 129,   0, 
      0,   0,   0,   0,   0,   0, 
     58, 128,  32,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      1,   0,   0,   7,  66,   0, 
     16,   0,   0,   0,   0,   0, 
     26,   0,  16,   0,   1,   0, 
      0,   0,  58,   0,  16,   0, 
      0,   0,   0,   0,  31,   0, 
      4,   3,  42,   0,  16,   0, 
      0,   0,   0,   0,  49,   0, 
      0,   7,  66,   0,  16,   0, 
      0,   0,   0,   0,   1,  64, 
      0,   0,  10, 215,  35,  60, 
     10,   0,  16,   0,   1,   0, 
      0,   0,  14,   0,   0,   8, 
    130,   0,  16,   0,   0,   0, 
      0,   0,  10,   0,  16,   0, 
      1,   0,   0,   0,  42, 128, 
     32,   0,   0,   0,   0,   0, 
      3,   0,   0,   0,   0,   0, 
      0,   8, 130,   0,  16,   0, 
      0,   0,   0,   0,  58,   0, 
     16, 128,  65,   0,   0,   0, 
      0,   0,   0,   0,   1,  64, 
      0,   0,   0,   0, 128,  63, 
     55,   0,   0,   9,  66,   0, 
     16,   0,   0,   0,   0,   0, 
     42,   0,  16,   0,   0,   0, 
      0,   0,  58,   0,  16,   0, 
      0,   0,   0,   0,   1,  64, 
      0,   0,   0,   0, 128,  63, 
    167,   0,   0, 139,   2,  99, 
      0, 128, 131, 153,  25,   0, 
     50,   0,  16,   0,   1,   0, 
      0,   0,  10,   0,  16,   0, 
      0,   0,   0,   0,   1,  64, 
      0,   0,   0,   0,   0,   0, 
     70, 224,  17,   0,   0,   0, 
      0,   0,  56,   0,   0,   7, 
     34,   0,  16,   0,   0,   0, 
      0,   0,  42,   0,  16,   0, 
      0,   0,   0,   0,  26,   0, 
     16,   0,   0,   0,   0,   0, 
      0,   0,   0,   7,  66,   0, 
     16,   0,   0,   0,   0,   0, 
     42,   0,  16,   0,   0,   0, 
      0,   0,  26,   0,  16,   0, 
      1,   0,   0,   0,  14,   0, 
      0,   7,  34,   0,  16,   0, 
      0,   0,   0,   0,  26,   0, 
     16,   0,   0,   0,   0,   0, 
     42,   0,  16,   0,   0,   0, 
      0,   0,  50,   0,   0,   9, 
     18,   0,  16,   0,   1,   0, 
      0,   0,  10,   0,  16,   0, 
      1,   0,   0,   0,  26,   0, 
     16,   0,   1,   0,   0,   0, 
     26,   0,  16,   0,   0,   0, 
      0,   0,  51,   0,   0,   7, 
     34,   0,  16,   0,   1,   0, 
      0,   0,  42,   0,  16,   0, 
      0,   0,   0,   0,   1,  64, 
      0,   0,   0,  80,  67,  71, 
     54,   0,   0,   5,  66,   0, 
     16,   0,   1,   0,   0,   0, 
      1,  64,   0,   0,   2,   0, 
      0,   0, 168,   0,   0,   9, 
    114, 224,  17,   0,   0,   0, 
      0,   0,  10,   0,  16,   0, 
      0,   0,   0,   0,   1,  64, 
      0,   0,   0,   0,   0,   0, 
     70,   2,  16,   0,   1,   0, 
      0,   0,  21,   0,   0,   1, 
     21,   0,   0,   1,  21,   0, 
      0,   1,  21,   0,   0,   1, 
     62,   0,   0,   1,  83,  84, 
     65,  84, 148,   0,   0,   0, 
     62,   0,   0,   0,   3,   0, 
      0,   0,   0,   0,   0,   0, 
      2,   0,   0,   0,  24,   0, 
      0,   0,  11,   0,   0,   0, 
      5,   0,   0,   0,   2,   0, 
      0,   0,   4,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      3,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   3,   0, 
      0,   0,   1,   0,   0,   0, 
      4,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      0,   0,   0,   0,   0,   0, 
      2,   0,   0,   0
};

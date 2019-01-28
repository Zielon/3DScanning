using UnityEngine;

namespace Assets.Scripts
{
    internal static class Helpers
    {
        public static Matrix4x4 GetPose(float[] pose)
        {
            var firstCol = new Vector4(pose[0], pose[1], pose[2], pose[3]);
            var secCol = new Vector4(pose[4], pose[5], pose[6], pose[7]);
            var thirdCol = new Vector4(pose[8], pose[9], pose[10], pose[11]);
            var fourthCol = new Vector4(pose[12], pose[13], pose[14], pose[15]);

            var cameraToWorld = new Matrix4x4();

            cameraToWorld.SetColumn(0, firstCol);
            cameraToWorld.SetColumn(1, secCol);
            cameraToWorld.SetColumn(2, thirdCol);
            cameraToWorld.SetColumn(3, fourthCol);

            var rot = cameraToWorld.rotation.eulerAngles;

            rot = new Vector3(rot.x, rot.y +180 ,rot.z );

            //   cameraToWorld.SetColumn(3, new Vector4(fourthCol.x, -fourthCol.z, fourthCol.y, 1));

            // Debug.Log(rot + "\n" + Quaternion.Euler(rot).eulerAngles.ToString());
            Debug.Log(cameraToWorld); 
            var ret = Matrix4x4.identity;
            ret.SetTRS(new Vector3(fourthCol.x, -fourthCol.y, fourthCol.z), Quaternion.Euler(rot), Vector3.one);
           // Debug.Log(cameraToWorld + "\n" + ret);

            return ret;
        }
    }
}
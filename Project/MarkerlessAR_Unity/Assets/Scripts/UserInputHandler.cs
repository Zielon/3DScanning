using UnityEngine;
using UnityEngine.SceneManagement;

public class UserInputHandler : MonoBehaviour
{
    public enum InputMode
    {
        IM_None,
        IM_SolidBall,
        IM_PaintBall,
        IM_PlaceObject,
        IM_FIRE
    }

    public InputMode currentMode = InputMode.IM_None;

    public GameObject FireSpawnerPrefab;
    public Camera firstPersonCamera;
    public GameObject FlameThrower;

    private bool mouseOverButton;
    public GameObject PaintBallPrefab;

    public GameObject placeObjectGhost;
    public GameObject PlaceObjectPrefab;

    public GameObject SolidBallPrefab;

    public float solidBallSpeed = 0.3f;
    public float spawnFireIntervall = 0.25f;
    private float spawnFireTimer;

    // Start is called before the first frame update
    private void Start()
    {
        placeObjectGhost.SetActive(false);
    }

    // Update is called once per frame
    private void Update()
    {
        RaycastHit hit;
        FlameThrower.SetActive(false);

        placeObjectGhost.SetActive(false);
        if (currentMode == InputMode.IM_PlaceObject &&
            Physics.Raycast(firstPersonCamera.ScreenPointToRay(Input.mousePosition), out hit, 1000))
            if (hit.collider.tag == "FrameMesh" || hit.collider.tag == "PlacedObject")
            {
                placeObjectGhost.SetActive(true);
                var collider = placeObjectGhost.GetComponent<Collider>();
                collider.enabled = true;
                placeObjectGhost.transform.position =
                    hit.point + hit.normal * Vector3.Dot(hit.normal, collider.bounds.extents) * 1.01f;
                placeObjectGhost.transform.up = hit.normal;
                Debug.DrawRay(hit.point, hit.normal * 1000);
                collider.enabled = false;
            }


        if (Input.GetMouseButtonDown(1)) //RMB
        {
            currentMode = InputMode.IM_None;
        }
        else if (Input.GetKey(KeyCode.F))
        {
            FlameThrower.SetActive(true);
            spawnFireTimer -= Time.deltaTime;

            if (spawnFireTimer < 0)
            {
                var o = Instantiate(FireSpawnerPrefab);
                o.transform.position = firstPersonCamera.transform.position;
                o.transform.forward = firstPersonCamera.ScreenPointToRay(Input.mousePosition).direction;

                spawnFireTimer = Random.Range(-0.75f, 2.0f) * spawnFireIntervall;
            }

            FlameThrower.transform.forward = firstPersonCamera.ScreenPointToRay(Input.mousePosition).direction;
            FlameThrower.transform.position = firstPersonCamera.transform.position +
                                              firstPersonCamera.ScreenPointToRay(Input.mousePosition).direction * 0.05f;
        }


        if (Input.GetMouseButtonDown(0) && !mouseOverButton) //LMB
            switch (currentMode)
            {
                case InputMode.IM_PaintBall:
                {
                    var o = Instantiate(PaintBallPrefab);
                    o.transform.position = firstPersonCamera.transform.position;
                    o.transform.forward = firstPersonCamera.ScreenPointToRay(Input.mousePosition).direction;


                    break;
                }
                case InputMode.IM_SolidBall:
                {
                    var o = Instantiate(SolidBallPrefab);
                    o.transform.position = firstPersonCamera.transform.position;
                    o.GetComponent<Rigidbody>().velocity =
                        firstPersonCamera.ScreenPointToRay(Input.mousePosition).direction * solidBallSpeed;
                    break;
                }
                case InputMode.IM_PlaceObject:
                {
                    if (placeObjectGhost.activeSelf) //obj can be placed in scene
                        Instantiate(PlaceObjectPrefab, placeObjectGhost.transform.position,
                            placeObjectGhost.transform.rotation);
                    break;
                }
            }
    }

    public void Menu()
    {
        SceneManager.LoadScene(0);
    }

    //Button args didnt work for some reason :x
    public void SetInputModeSolidBall()
    {
        currentMode = InputMode.IM_SolidBall;
    }

    public void SetInputModePaintBall()
    {
        currentMode = InputMode.IM_PaintBall;
    }

    public void SetInputModePlaceObject()
    {
        currentMode = InputMode.IM_PlaceObject;
    }

    public void buttonMouseOverEnter()
    {
        mouseOverButton = true;
    }

    public void buttonMouseOverExit()
    {
        mouseOverButton = false;
    }
}
Workflow: 

build DLL 
move DLL to .Assets/Plugins (Unity may need to be closed for this) 
call functions from unity scripts



The scene: 
The camera has a sprite (videoBG) attached which will always be located at the far plane. 
This sprite also contains a script called dllInteract.cs the idea of this script is to read images via the DLL (size is hardcoded atm) and set the sprite to the read image. 

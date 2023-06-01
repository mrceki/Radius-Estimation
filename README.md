# Object Radius Estimation with YOLOv7 and Depth Camera

This repository uses the output of YOLOv7 along with a depth camera to estimate the radius of detected objects. It is primarily used in the following repository: [https://github.com/mrceki/air_robot_arm](https://github.com/mrceki/air_robot_arm).

The purpose of this method is to enable a humanoid robot arm to perform pick-and-place tasks. Normally, methods like 6D pose estimation are used for this purpose. However, in this approach, you can train a single object in a single dimension specific to the model, making it costly and time-consuming to create a dataset, train, and repeat these steps for a new object. With this method, you can add any object that can be detected by CNN to the robot's planning world and perform the necessary pick-and-place tasks with millimeter precision. Additionally, since the cost is low, environment refreshing for collision avoidance is performed frequently. The system is built on ROS, making it highly flexible and easy to use with other CNN models and depth cameras.

![Başlıksız Diyagram drawio (1)](https://github.com/mrceki/Radius-Estimation/assets/105711013/5afb0d5c-ff76-40fa-991b-a0620d27f467)


![ezgif com-video-to-gif](https://github.com/mrceki/Radius-Estimation/assets/105711013/77234594-b20a-4c15-9fe3-3b104b2d62df)

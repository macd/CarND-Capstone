# Self Driving Car Nano-Degree Capstone Project Repository

This is the personal project repo of Don MacMillen for the final
project of the Udacity Self-Driving Car Nanodegree: Programming a Real
Self-Driving Car. For more information about the project, see the
project introduction
[here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

##Team Members:   
<center>

| Name | Contact | Location |
| ------------- |:-------------| :-----|
| Eric Kim | ericjkim9@gmail.com | Santa Clara, CA |
| Don MacMillen | don.macmillen@gmail.com | San Mateo, CA |
| Grigory Makarevich | grigorymakarevich@gmail.com | Seattle, WA |
| Stefan Gantner	| stefangantner@live.de | Munich, Germany |
| Karsten Schwinne | kschwinne@gmail.com | Dortmund, Germany |

</center>

## More info
If you are curious, more info can be found in the final_report.md file

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) that was recorded on the Udacity self-driving car (a bag demonstraing the correct predictions in autonomous mode can be found [here](https://drive.google.com/open?id=0B2_h37bMVw3iT0ZEdlF4N01QbHc))
2. Unzip the file
```bash
unzip traffic_light_bag_files.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_files/loop_with_traffic_light.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Docker GPU instance build and use for Hacker-force


### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

### GPU/CUDA support
[Install nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

Build the docker container. Note that if you build the image in the
project repo, it will populate with all of the repo's files so you
will not have to clone after you log into it the first time.  Also,
the Dockerfile.gpu will install all of the requirements (well mostly,
see below)

```bash
nvidia-docker build . -f Dockerfile.gpu -t capstone-gpu
```

### run-cuda script

You can simply start the container or attach to it using the run-cuda or run-devel-cuda script
```bash
./run-cuda.sh
```

Note that this solution, borrowed from team-inrs, is just a little bit
on the hacky side.  Everything is in a top level /udacity directory and
everything is run as root.  I made no attempt to clean this up.

Also, Eric, for your stuff I had to clone, build and install protobuf.
Clone the google models, run protoc on some of the models, pip install
matplotlib, apt-get install python-tk, and finally rosdep update.

It sounds like much of this may not be necessary with Grigory's approach.

### Networking

When using the run-cuda.sh command, it does the Docker version of port
forwarding from docker container port 4567 to host port 4567.  I
didn't chase down how to do a bridged to the router thing for the
docker container, because I am running the simulator on another
computer.  If you are running on another computer, you now do the port
forwarding there to the ip address of the (docker) host and not of the
docker container.  So this probably won't work well on a single
computer with both the simulator and ROS inside of the container
approach.



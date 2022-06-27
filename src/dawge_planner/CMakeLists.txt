cmake_minimum_required(VERSION 3.0.2)
project(dawge_planner)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  unitree_legged_msgs
  # unitree_legged_real
)

# Run the python setup.py -> Helps with importing modules from this package
catkin_python_setup() 

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs unitree_legged_msgs unitree_legged_real
)

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${Boost_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "-O3")

# In order to use UNITREE_LEGGED_SDK namespace we should add these to CMakeLists.txt
include_directories($ENV{UNITREE_LEGGED_SDK_PATH}/include)
link_directories($ENV{UNITREE_LEGGED_SDK_PATH}/lib)
string(CONCAT LEGGED_SDK_NAME libunitree_legged_sdk_$ENV{UNITREE_PLATFORM}.so)
set(EXTRA_LIBS ${LEGGED_SDK_NAME} lcm)
add_definitions(-DSDK3_2)

# TODO: Uncomment these on the actual robot
# add_library(leg_controller_lib src/LegController.cpp)
# add_dependencies(leg_controller_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# add_executable(main_helper src/exe/main_helper.cpp src/LegController.cpp)
# target_link_libraries(main_helper leg_controller_lib ${EXTRA_LIBS} ${catkin_LIBRARIES})
# add_dependencies(main_helper ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# In this computer this part should be commented out
# add_executable(comm_hw src/exe/comm_hardware.cpp)
# target_link_libraries(comm_hw ${EXTRA_LIBS} ${catkin_LIBRARIES})
# add_dependencies(comm_hw ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

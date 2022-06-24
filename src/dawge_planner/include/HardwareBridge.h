// Class to connect high level task requirements to low level lcm messages
// This class will listen HighCmd messages and publish HighState messages
// and then it converts them to LCM messages

#include <ros/ros.h>
#include <string>
#include <unitree_legged_msgs/HighCmd.h>
#include <unitree_legged_msgs/HighState.h>
#include "unitree_legged_real/convert.h

using namespace UNITREE_LEGGED_SDK;

class HardwareBridge
{
  public: 
    HardwareBridge(ros::NodeHandle nh, std::string highCmdTopic, std::string highStateTopic);

    void run(); // Main function to run - this will start a loop 

    UNITREE_LEGGED_SDK::HighCmd getLcmHighCmd() { return _lcmHighCmd; }
    void setlcmHighState(UNITREE_LEGGED_SDK::HighState* lcmHighStatePtr) {
      &_lcmHighState = lcmHighStatePtr; // TODO: make sure that this works somewhat 
    } 

  private:
    
    void highCmdCallback(const unitree_legged_msgs::HighCmd& msg);

    ros::NodeHandle _n;
    ros::Publisher _pub;
    ros::Subscriber _sub;
    ros::Rate _r;

    unitree_legged_msgs::HighCmd _rosHighCmd;
    unitree_legged_msgs::HighState _rosHighState;
    UNITREE_LEGGED_SDK::HighCmd _lcmHighCmd; // inside the callback high cmd ros messages will be converted to lcm messages and will be able to retrieved from outer class
    UNITREE_LEGGED_SDK::HighState _lcmHighState; // this will be received from an outer helper class and then converted and published as a ROS message

}
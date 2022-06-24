// Executable to communicate with the hardware of the robot - similar to main_helper and walk_mode 
// It listens to lcm messages from robot and high commands from python nodes 
// with task reqiurements.

#include <ros/ros.h>
#include <string>
#include <pthread.h>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <unitree_legged_msgs/HighCmd.h>
#include <unitree_legged_msgs/HighState.h>
#include "unitree_legged_real/convert.h"

using namespace UNITREE_LEGGED_SDK;

template<typename TLCM>
void* update_loop(void* param) {
    TLCM *data = (TLCM *)param;
    while(ros::ok){
        data->Recv();
        usleep(2000);
    }
}

UNITREE_LEGGED_SDK::HighCmd highCmdLCM;
void highCmdROSCallback(unitree_legged_msgs::HighCmd& highCmdROS) {
    printf("received highCmdROS in high cmd callback: %s", highCmdROS.data);
    highCmdLCM = ToLcm(highCmdROS, highCmdLCM);
}

// TODO: This should be done in a ROS class!!
int mainHelper(int argc, char *argv[],
               UNITREE_LEGGED_SDK::LCM &roslcm,
               std::string highStateTopic, std::string highCmdTopic) {

    // Set up the helper warning messages
    std::cout << "WARNING: Control level is set to HIGH-level." << std::endl
              << "Make sure the robot is standing on the ground." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    // ROS initializations
    ros::NodeHandle n;
    ros::Rate loopRate(20); // TODO: This if this is too low
    ros::Publisher pub;
    ros::Subscriber sub;

    UNITREE_LEGGED_SDK::HighState highStateLCM = {0}; // Will be reveived through lcm - ROS msg will be published
    unitree_legged_msgs::HighState highStateROS; 

    roslcm.SubscribeState();

    pthread_t tid; 
    pthread_create(&tid, NULL, update_loop<TLCM>, &roslcm);

    // Initialize ROS publishers/subscribers
    printf("highStateTopic: %s", highStateTopic);
    pub = n.advertise<unitree_legged_msgs::HighState>(highStateTopic, 1000);
    sub = n.subscribe(highCmdTopic, 1000, highCmdROSCallback);

    // Get the first message if there is any
    ros::spinOnce();

    // Run the loop - send high commands to lcm 
    // publish the highStateROS
    while (ros::ok()) {
        // Get the LCM high state
        roslcm.Get(highStateLCM);
        highStateROS = ToRos(highStateLCM);

        // Publish high state
        pub.publish(highStateROS);

        // Send the highCommandLCM
        roslcm.Send(highCmdLCM);

        // Sleep and spinOnce
        ros::spinOnce();
        loopRate.sleep();

    }
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "dawge_comm_hw_highlevel");

    std::string highStateTopic = "dawge_high_state";
    std::string highCmdTopic = "dawge_high_cmd";

    UNITREE_LEGGED_SDK::LCM roslcm(UNITREE_LEGGED_SDK::HIGHLEVEL);
    mainHelper(argc, argv, roslcm, highStateTopic, highCmdTopic);

}
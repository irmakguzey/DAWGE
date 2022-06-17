#include <ros/ros.h>
#include <string>
#include <pthread.h>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <unitree_legged_msgs/LowCmd.h>
#include <unitree_legged_msgs/LowState.h>
#include "convert.h"

using namespace UNITREE_LEGGED_SDK;

// NOTE: This file should be inserted inside unitree_legged_real/src/exe/

// Loop to get data from lcm server
template<typename TLCM>
void* update_loop(void* param)
{
    TLCM *data = (TLCM *)param;
    while(ros::ok){
        data->Recv();
        usleep(2000);
    }
}

double jointLinearInterpolation(double initPos, double targetPos, double rate)
{
    double p;
    rate = std::min(std::max(rate, 0.0), 1.0);
    // Will make the robot go to the target pose slowly
    p = initPos*(1-rate) + targetPos*rate;
    return p;
}

template<typename TCmd, typename TState, typename TLCM>
int mainHelper(int argc, char *argv[], TLCM &roslcm)
{
    std::cout << "DAWGE LOW LEVEL CONTROL" << std::endl
              << "Make sure the robot is hung up." << std::endl 
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    ros::NodeHandle n;
    ros::Rate loop_rate(500); // 500Hz frequency to send the data

    int rate_count = 0; // To make the joints move slowly -- NOTE: should be set to zero when there is a new end effector command
    float qInit[3] = {0};
    float qDes[3] = {0};
    float goalQs[3] = {0.0, 1.2, -2.0};
    float Kp[3] = {5.0, 5.0, 5.0};
    float Kd[3] = {1.0, 1.0, 1.0};
    TCmd SendLowLCM = {0};
    TState RecvLowLCM = {0};
    unitree_legged_msgs::LowCmd SendLowROS;
    unitree_legged_msgs::LowState RecvLowROS; 

    bool initiated_flag = false; // We will wait until count reached to 10 to send commands
    int count = 0;

    roslcm.SubscribeState(); 

    pthread_t tid;
    pthread_create(&tid, NULL, update_loop<TLCM>, &roslcm);

    // Set the message sending level to low
    // SendLowROS.levelFlag = LOWLEVEL;
    printf("levelFlag set to LOWLEVEL");

    while (ros::ok()) {
        roslcm.Get(RecvLowLCM);
        RecvLowROS = ToRos(RecvLowLCM);
        // Printing torque positions of each joint - let's not modify the torques at all
        printf("Hip Torques: [FR_0:%f, FL_0:%f, RR_0:%f, RL_0:%f]\n",
               RecvLowROS.motorState[FR_0].tauEst,
               RecvLowROS.motorState[FL_0].tauEst,
               RecvLowROS.motorState[RR_0].tauEst,
               RecvLowROS.motorState[RL_0].tauEst);


        // Set the torques to be a bit larger than current state
        for(int i = 0; i<12; i++){
            if (RecvLowROS.motorState[i].tauEst < 0) {
                SendLowROS.motorCmd[i].tau = RecvLowROS.motorState[i].tauEst - 0.65f;
            } else {
                SendLowROS.motorCmd[i].tau = RecvLowROS.motorState[i].tauEst + 0.65f;
            }
        }
        if (initiated_flag == true) {

            // Get the initial positions of front right arm
            qInit[0] = RecvLowROS.motorState[FR_0].q;
            qInit[1] = RecvLowROS.motorState[FR_1].q;
            qInit[2] = RecvLowROS.motorState[FR_2].q;

            rate_count++;
            double rate = rate_count / 500.0; // It will move to the desired position slowly

            // Set the destionation positions by using jointLinearInterpolation
            qDes[0] = jointLinearInterpolation(qInit[0], goalQs[0], rate);
            qDes[1] = jointLinearInterpolation(qInit[1], goalQs[1], rate);
            qDes[2] = jointLinearInterpolation(qInit[2], goalQs[2], rate);

            // Set the values in the created message
            SendLowROS.motorCmd[FR_0].q = qDes[0];
            SendLowROS.motorCmd[FR_0].dq = 0;
            SendLowROS.motorCmd[FR_0].Kp = Kp[0];
            SendLowROS.motorCmd[FR_0].Kd = Kd[0];
            // SendLowROS.motorCmd[FR_0].tau = -0.65f;

            SendLowROS.motorCmd[FR_1].q = qDes[1];
            SendLowROS.motorCmd[FR_1].dq = 0;
            SendLowROS.motorCmd[FR_1].Kp = Kp[1];
            SendLowROS.motorCmd[FR_1].Kd = Kd[1];
            // SendLowROS.motorCmd[FR_1].tau = 0.0f;

            SendLowROS.motorCmd[FR_2].q =  qDes[2];
            SendLowROS.motorCmd[FR_2].dq = 0;
            SendLowROS.motorCmd[FR_2].Kp = Kp[2];
            SendLowROS.motorCmd[FR_2].Kd = Kd[2];
            // SendLowROS.motorCmd[FR_2].tau = 0.0f;

        }

        // SendLowLCM = ToLcm(SendLowROS, SendLowLCM);
        // roslcm.Send(SendLowLCM); - NOTE: let's not send it for now
        ros::spinOnce();
        loop_rate.sleep();

        count++;
        if(count > 10){
            count = 10;
            initiated_flag = true;
        }
    }
    return 0;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "position_ros_mode");
    std::string firmwork;
    ros::param::get("/firmwork", firmwork);

    std::string robot_name;
    UNITREE_LEGGED_SDK::LeggedType rname;
    ros::param::get("/robot_name", robot_name);
    if(strcasecmp(robot_name.c_str(), "A1") == 0)
        rname = UNITREE_LEGGED_SDK::LeggedType::A1;
    else if(strcasecmp(robot_name.c_str(), "Aliengo") == 0)
        rname = UNITREE_LEGGED_SDK::LeggedType::Aliengo;
        
    UNITREE_LEGGED_SDK::LCM roslcm(LOWLEVEL);
    mainHelper<UNITREE_LEGGED_SDK::LowCmd, UNITREE_LEGGED_SDK::LowState, UNITREE_LEGGED_SDK::LCM>(argc, argv, roslcm);
}
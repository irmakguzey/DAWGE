// Class to put it all together and update the data in a thread
// TLCM message will be passed as an argument to updateData method of LegController
// On top of the RobotRunner we will use one more wrapper - for simulation or for real life
// For now simulation wrapper will not be used

#include <ros/ros.h>
#include <string>
#include <pthread.h>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <unitree_legged_msgs/LowCmd.h>
#include <unitree_legged_msgs/LowState.h>
#include "unitree_legged_real/convert.h"

#include <LegController.h> // LegController Header - our own leg controller implementation

using namespace UNITREE_LEGGED_SDK;

// Method to retrieve data from LCM all the time - this is called in a thread 
template<typename TLCM>
void* update_loop(void* param)
{
    TLCM *data = (TLCM *)param;
    while(ros::ok){
        data->Recv();
        usleep(2000);
    }
}

// Positions to send might be sent from this node
float jointLinearInterpolation(float initPos, float targetPos, float rate)
{
    float p;
    rate = std::min(std::max(rate, 0.0f), 1.0f);
    p = initPos*(1-rate) + targetPos*rate;
    return p;
}

template<typename TCmd, typename TState, typename TLCM>
int mainHelper(int argc, char *argv[], TLCM &roslcm)
{
    // Set up the helper warning messages
    std::cout << "WARNING> Control level is set to LOW-level." << std::endl
              << "Make sure the robot is hung up." << std::endl
              << "Press Enter to continue..." << std::endl; 

    std::cin.ignore();

    // ROS initializations
    ros::NodeHandle n;
    ros::Rate loop_rate(500);

    long motiontime = 0;
    int rate_count = 0; // Rate to give to jointLinearInterpolation - makes a smoother move
    bool initiated_flag = false; // Give time for the initialization
    int count = 0;

    // Initialize the standing target joint positions
    float standing_target_jpos[4][3] =
    {
        {-0.077036, 0.781234, -1.678993}, // FR
        {0.077036, 0.781234, -1.678993}, // FL
        {-0.077036, 0.781234, -1.678993}, // RR
        {0.077036, 0.781234, -1.678993}, // RL
    };

    LegController<float> legController;
    // UNITREE_LEGGED_SDK::LCM roslcm(LOWLEVEL);

    TCmd sendLowLCM = {0};
    TState recvLowLCM = {0};
    unitree_legged_msgs::LowCmd sendLowROS;
    unitree_legged_msgs::LowState recvLowROS;

    roslcm.SubscribeState();

    pthread_t tid;
    pthread_create(&tid, NULL, update_loop<TLCM>, &roslcm);

    sendLowROS.levelFlag = LOWLEVEL;
    // Make the robot lay down and get into low level mode
    for(int i = 0; i<12; i++){
        sendLowROS.motorCmd[i].mode = 0x0A;   // motor switch to servo (PMSM) mode
        sendLowROS.motorCmd[i].q = PosStopF;
        sendLowROS.motorCmd[i].Kp = 0;
        sendLowROS.motorCmd[i].dq = VelStopF;
        sendLowROS.motorCmd[i].Kd = 0;
        sendLowROS.motorCmd[i].tau = 0;
    }

    while(ros::ok()) {
        roslcm.Get(recvLowLCM);
        recvLowROS = ToRos(recvLowLCM); // Use this for the setup and final steps of leg controller

        // Update the data in LegController
        legController.updateData(recvLowROS);

        if (initiated_flag == true) {
            motiontime++;    

            // Initial position of each feet are calculated in updateData (in computeLegJacobianAndPosition)        
            std::vector< Vec3<float> > ini_feet_pos;
            Vec3<float> ini_leg_pos; // Initial joint angles for one leg (traversed through all legs in setting commands)
            

            // Set the joint kp/kds
            Mat3<float> kpJointMat;
            Mat3<float> kdJointMat;
            Mat3<float> kpCartesianMat;
            Mat3<float> kdCartesianMat;
            // Update the jpos feedback gains - NOTE: Using same kpMat/kdMat as MIT cheetah robot
            kpJointMat << 50, 0, 0,
                     0, 50, 0,
                     0, 0, 50;
            kdJointMat << 1, 0, 0,
                     0, 1, 0,
                     0, 0, 1;
                
            // Set kp/kd Cartesians and pDes for standing up - this is only for standing up 
            // TODO: There should be a class for each task (there could even be a super class named task)
            // And cartesian gains and p/vs of the legcontroller commands should be set in those tasks
            kpCartesianMat << 500, 0, 0, // TODO: These are taken from FSM_State_Standup.cpp
                              0, 500, 0,
                              0, 0, 500;
            kdCartesianMat << 8, 0, 0,
                              0, 8, 0,
                              0, 0, 8;
            for (int leg = 0; leg < 4; leg++) {
                legController.commands[leg].kpJoint = kpJointMat; 
                legController.commands[leg].kdJoint = kdJointMat;
                legController.commands[leg].kpCartesian = kpCartesianMat;
                legController.commands[leg].kdCartesian = kdCartesianMat;
                
                // Set the desired position for each feet (for standing up)
                // TODO: Delete these parts for now? 
                ini_feet_pos[leg] = legController.datas[leg].p;
                legController.commands[leg].pDes = ini_feet_pos[leg];
                legController.commands[leg].pDes[2] = jointLinearInterpolation(ini_feet_pos[leg][2], -0.3, motiontime/200.0);
            
                // Set the desired angle positions for each joint
                // Get the current joint position
                ini_leg_pos = legController.datas[leg].q;
                // Get desired joint angle for each joint
                for (int jid = 0; jid < 3; jid++) {
                    legController.commands[leg].qDes[jid] = jointLinearInterpolation(ini_leg_pos[jid], 
                                                                                     standing_target_jpos[leg][jid],
                                                                                     motiontime/200.0);
                }
            }
            // this->_data->_legController->commands[i].pDes = _ini_foot_pos[i];
            // this->_data->_legController->commands[i].pDes[2] = 
            //     progress*(-hMax) + (1. - progress) * _ini_foot_pos[i][2]; -> -hMax olduguna dikkat et!! 
        }

        // update the command in leg controller
        legController.updateCommand(sendLowROS); // TODO: add main_helper.cpp inside the robot

        // Set up 
        sendLowLCM = ToLcm(sendLowROS, sendLowLCM);
        roslcm.Send(sendLowLCM);
        ros::spinOnce();
        loop_rate.sleep();

        count++;
        if(count > 10){
            count = 10;
            initiated_flag = true;
        }
    }
}

int main(int argc, char *argv[]){
    ros::init(argc, argv, "dawge_main_helper_lowlevel");

    UNITREE_LEGGED_SDK::LCM roslcm(LOWLEVEL);
    mainHelper<UNITREE_LEGGED_SDK::LowCmd, UNITREE_LEGGED_SDK::LowState, UNITREE_LEGGED_SDK::LCM>(argc, argv, roslcm);
}
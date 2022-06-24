#include "HardwareBridge.h"

HardwareBridge::HardwareBridge(ros::NodeHandle nh, std::string highCmdTopic, std::string highStateTopic) {
    // Set up publishers and subscribers
    _n = nh;
    _sub = _n.subscribe(highCmdTopic, 1000, &HardwareBridge::highCmdCallback, this);
    _pub = _n.advertive<unitree_legged_msgs::HighState>(highStateTopic, 1000);

}

HardwareBridge::run() {
    // TODO
}

HardwareBridge::highCmdCallback(const unitree_legged_msgs::HighCmd& msg) {
    
}


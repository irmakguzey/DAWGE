#include <ros/ros.h>
#include <string>
#include <pthread.h> 
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp> 
#include <unitree_legged_msgs/LowCmd.h> 
#include <unitree_legged_msgs/LowState.h>
#include "convert.h" // This should be added to this package
#include "cppTypes.h"


using namespace UNITREE_LEGGED_SDK;

template <typename T>
struct LegControllerCommand {
  LegControllerCommand() { zero(); }

  void zero();

  Vec3<T> tauFeedForward, forceFeedForward, qDes, qdDes, pDes, vDes;
  Mat3<T> kpCartesian, kdCartesian, kpJoint, kdJoint;
};

template <typename T>
struct LegControllerData {
  LegControllerData() { zero(); }

  void zero();

  Vec3<T> q, qd, p, v;
  Mat3<T> J; // Jacobian - will be used in the future
  Vec3<T> tauEstimate;
};

template <typename T>
class LegController {
 public:
  LegController() { zeroCommand(); }

  void zeroCommand();
  // void edampCommand(RobotType robot, T gain); -- TODO: could be implemented later
  void updateData(unitree_legged_msgs::LowState &recvLowROS);
  void updateCommand(unitree_legged_msgs::LowCmd &sendLowROS);
  // void setEnabled(bool enabled) { _legsEnabled = enabled; };
  // void setLcm(leg_control_data_lcmt* data, leg_control_command_lcmt* command);


  LegControllerCommand<T> commands[4];
  LegControllerData<T> datas[4];
  // bool _legsEnabled = false; -- These are not used - yet
  // bool _zeroEncoders = false;
  // u32 _calibrateEncoders = 0;
};

template <typename T>
void computeLegJacobianAndPosition(Vec3<T>& q, Mat3<T>* J,
                                   Vec3<T>* p, int leg);

#endif  // PROJECT_LEGCONTROLLER_H

#include "LegController.h"

template <typename T>
void LegControllerCommand<T>::zero() {
  tauFeedForward = Vec3<T>::Zero();
  forceFeedForward = Vec3<T>::Zero();
  qDes = Vec3<T>::Zero();
  qdDes = Vec3<T>::Zero();
  pDes = Vec3<T>::Zero();
  vDes = Vec3<T>::Zero();
  kpCartesian = Mat3<T>::Zero();
  kdCartesian = Mat3<T>::Zero();
  kpJoint = Mat3<T>::Zero();
  kdJoint = Mat3<T>::Zero();
}

template <typename T>
void LegControllerData<T>::zero() {
  q = Vec3<T>::Zero();
  qd = Vec3<T>::Zero();
  p = Vec3<T>::Zero();
  v = Vec3<T>::Zero();
  J = Mat3<T>::Zero();
  tauEstimate = Vec3<T>::Zero();
}

template <typename T>
void LegController<T>::zeroCommand() {
  for (auto& cmd : commands) {
    cmd.zero();
  }
}


// Methods in LegController for data and command they only use ROS states and commands
template <typename T>
void LegController<T>::updateData(unitree_legged_msgs::LowState &recvLowROS) {

  // RecvLowROS = ToRos(RecvLowLCM) is considered to already be gone to RecvLowROS has 
  // robot data currently
  for (int leg = 0; leg < 4; leg++) {
    for (int jid = 0; jid < 3; jid++) {
      // Set the joint positions
      datas[leg].q(jid) = recvLowROS.motorState[3*leg + jid].q;
      // Set the joint velocities
      datas[leg].qd(jid) = recvLowROS.motorState[3*leg + jid].dq;
    }

    // Compute Jacobian and Position
    // TODO: Implement this method
    computeLegJacobianAndPosition<T>(datas[leg].q, &(datas[leg].J), &(datas[leg].p), leg); 

    datas[leg].v = datas[leg].J * datas[leg].qd;
  }
}

// TODO: You should check if sendLowROS changes at the end of the function
template <typename T>
void LegController<T>::updateCommand(unitree_legged_msgs::LowCmd &sendLowROS) {
  for (int leg = 0; leg < 4; leg++) {
    // tauFF
    Vec3<T> legTorque = commands[leg].tauFeedForward; // Get the previous given command

    // foot force
    Vec3<T> footForce = commands[leg].forceFeedForward; // They get updated soon

    // cartesian PD - TODO: kpCartesian/kdCartesian should be set in mainHelper according to the task
    footForce += commands[leg].kpCartesian * (commands[leg].pDes - datas[leg].p);
    footForce += commands[leg].kdCartesian * (commands[leg].vDes - datas[leg].v);

    // calculate the leg torque
    legTorque += datas[leg].J.transpose() * footForce;

    // set the ROS commands
    for (int jid = 0; jid < 3; jid++) {
      // command torque
      sendLowROS.motorCmd[3*leg + jid].tau = legTorque(jid);

      // joint space PD
      sendLowROS.motorCmd[3*leg + jid].Kd = commands[leg].kdJoint(jid, jid);
      sendLowROS.motorCmd[3*leg + jid].Kp = commands[leg].kpJoint(jid, jid);

      // destinated position and velocity
      sendLowROS.motorCmd[3*leg + jid].q = commands[leg].qDes(jid);
      sendLowROS.motorCmd[3*leg + jid].dq = commands[leg].qdDes(jid);
      
    }
    // Estimate torque -- I think this is torque we get from the robot (?)
    datas[leg].tauEstimate = legTorque +
                             commands[leg].kpJoint * (commands[leg].qDes - datas[leg].q) +
                             commands[leg].kdJoint * (commands[leg].qdDes - datas[leg].qd);
  }
}

template struct LegControllerCommand<double>;
template struct LegControllerCommand<float>;

template struct LegControllerData<double>;
template struct LegControllerData<float>;

template class LegController<double>;
template class LegController<float>;

/*!
 * Compute the position of the foot and its Jacobian.  This is done in the local
 * leg coordinate system. If J/p are NULL, the calculation will be skipped.
 */
template <typename T>
void computeLegJacobianAndPosition(Vec3<T>& q, Mat3<T>* J,
                                   Vec3<T>* p, int leg) {
  T l1 = 0.04f; // Got FR_hip link length from urdf
  T l2 = 0.2f; // Got FR_thight link size.x from urdf
  T l3 = 0.2f; // Got FR_calf link size.x from urdf
  T l4 = 0.0f; // I gave it 0.0 bc of the original values from MIT - not sure if these are true

  // Followings are the MIT constants that is originally used - could be used afterwards
  // cheetah._abadLinkLength = 0.045;
  // cheetah._hipLinkLength = 0.342;
  // cheetah._kneeLinkY_offset = 0.0;
  // cheetah._kneeLinkLength = 0.345;
  
  // Get if the i-th leg is on the left (+) or right (-) of the robot.
  // The side sign (-1 for right legs, +1 for left legs)
  const T sideSigns[4] = {-1, 1, -1, 1}; // Order is set from unitree_sdk constants for each joint index
  T sideSign = sideSigns[leg];

  T s1 = std::sin(q(0));
  T s2 = std::sin(q(1));
  T s3 = std::sin(q(2));

  T c1 = std::cos(q(0));
  T c2 = std::cos(q(1));
  T c3 = std::cos(q(2));

  T c23 = c2 * c3 - s2 * s3;
  T s23 = s2 * c3 + c2 * s3;

  if (J) {
    J->operator()(0, 0) = 0;
    J->operator()(0, 1) = l3 * c23 + l2 * c2;
    J->operator()(0, 2) = l3 * c23;
    J->operator()(1, 0) = l3 * c1 * c23 + l2 * c1 * c2 - (l1+l4) * sideSign * s1;
    J->operator()(1, 1) = -l3 * s1 * s23 - l2 * s1 * s2;
    J->operator()(1, 2) = -l3 * s1 * s23;
    J->operator()(2, 0) = l3 * s1 * c23 + l2 * c2 * s1 + (l1+l4) * sideSign * c1;
    J->operator()(2, 1) = l3 * c1 * s23 + l2 * c1 * s2;
    J->operator()(2, 2) = l3 * c1 * s23;
  }

  if (p) {
    p->operator()(0) = l3 * s23 + l2 * s2;
    p->operator()(1) = (l1+l4) * sideSign * c1 + l3 * (s1 * c23) + l2 * c2 * s1;
    p->operator()(2) = (l1+l4) * sideSign * s1 - l3 * (c1 * c23) - l2 * c1 * c2;
  }
}

template void computeLegJacobianAndPosition<double>(Quadruped<double>& quad,
                                                    Vec3<double>& q,
                                                    Mat3<double>* J,
                                                    Vec3<double>* p, int leg);
template void computeLegJacobianAndPosition<float>(Quadruped<float>& quad,
                                                   Vec3<float>& q,
                                                   Mat3<float>* J,
                                                   Vec3<float>* p, int leg);

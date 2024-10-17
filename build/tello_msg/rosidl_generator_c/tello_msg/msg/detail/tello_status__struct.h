// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from tello_msg:msg/TelloStatus.idl
// generated code does not contain a copyright notice

#ifndef TELLO_MSG__MSG__DETAIL__TELLO_STATUS__STRUCT_H_
#define TELLO_MSG__MSG__DETAIL__TELLO_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'acceleration'
// Member 'speed'
#include "geometry_msgs/msg/detail/vector3__struct.h"
// Member 'wifi_snr'
#include "rosidl_runtime_c/string.h"

// Struct defined in msg/TelloStatus in the package tello_msg.
typedef struct tello_msg__msg__TelloStatus
{
  geometry_msgs__msg__Vector3 acceleration;
  geometry_msgs__msg__Vector3 speed;
  int32_t barometer;
  int32_t distance_tof;
  uint8_t battery;
  int32_t fligth_time;
  int32_t highest_temperature;
  int32_t lowest_temperature;
  float temperature;
  int32_t pitch;
  int32_t roll;
  int32_t yaw;
  rosidl_runtime_c__String wifi_snr;
} tello_msg__msg__TelloStatus;

// Struct for a sequence of tello_msg__msg__TelloStatus.
typedef struct tello_msg__msg__TelloStatus__Sequence
{
  tello_msg__msg__TelloStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} tello_msg__msg__TelloStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // TELLO_MSG__MSG__DETAIL__TELLO_STATUS__STRUCT_H_

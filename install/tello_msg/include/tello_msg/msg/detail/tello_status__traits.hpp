// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tello_msg:msg/TelloStatus.idl
// generated code does not contain a copyright notice

#ifndef TELLO_MSG__MSG__DETAIL__TELLO_STATUS__TRAITS_HPP_
#define TELLO_MSG__MSG__DETAIL__TELLO_STATUS__TRAITS_HPP_

#include "tello_msg/msg/detail/tello_status__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'acceleration'
// Member 'speed'
#include "geometry_msgs/msg/detail/vector3__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<tello_msg::msg::TelloStatus>()
{
  return "tello_msg::msg::TelloStatus";
}

template<>
inline const char * name<tello_msg::msg::TelloStatus>()
{
  return "tello_msg/msg/TelloStatus";
}

template<>
struct has_fixed_size<tello_msg::msg::TelloStatus>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<tello_msg::msg::TelloStatus>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<tello_msg::msg::TelloStatus>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TELLO_MSG__MSG__DETAIL__TELLO_STATUS__TRAITS_HPP_

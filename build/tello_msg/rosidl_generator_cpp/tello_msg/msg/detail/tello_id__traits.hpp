// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tello_msg:msg/TelloID.idl
// generated code does not contain a copyright notice

#ifndef TELLO_MSG__MSG__DETAIL__TELLO_ID__TRAITS_HPP_
#define TELLO_MSG__MSG__DETAIL__TELLO_ID__TRAITS_HPP_

#include "tello_msg/msg/detail/tello_id__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<tello_msg::msg::TelloID>()
{
  return "tello_msg::msg::TelloID";
}

template<>
inline const char * name<tello_msg::msg::TelloID>()
{
  return "tello_msg/msg/TelloID";
}

template<>
struct has_fixed_size<tello_msg::msg::TelloID>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<tello_msg::msg::TelloID>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<tello_msg::msg::TelloID>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TELLO_MSG__MSG__DETAIL__TELLO_ID__TRAITS_HPP_

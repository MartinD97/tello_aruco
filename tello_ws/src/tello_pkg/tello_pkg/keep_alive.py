import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class CommandNode(Node):
    def __init__(self):
        super().__init__('command_node')
        self.pub_control = self.create_publisher(Twist, 'control', 10)
        self.sub_command = self.create_subscription(Twist, 'control', self.command_callback, 10)
        self.timer_period = 6.0
        self.last_command_time = time.time()
        self.timer = self.create_timer(self.timer_period, self.send_keep_alive_if_needed)

    def command_callback(self, msg):
        self.last_command_time = time.time()
        self.get_logger().info('Received command, resetting timer.')

    def send_keep_alive_if_needed(self):
        current_time = time.time()
        if (current_time - self.last_command_time) >= self.timer_period:
            msg = Twist()
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.z = 0.0
            self.pub_control.publish(msg)
            self.get_logger().info('Keep-alive command sent to drone.')

def main(args=None):
    rclpy.init(args=args)
    command_node = CommandNode()
    rclpy.spin(command_node)
    command_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

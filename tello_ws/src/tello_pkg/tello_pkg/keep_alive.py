import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CommandNode(Node):
    def __init__(self):
        super().__init__('command_node')
        self.pub_control = self.create_publisher(Twist, 'control', 10)
        self.timer_period = 6.0
        self.timer = self.create_timer(self.timer_period, self.send_command)

    def send_command(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.z = 0.0

        self.pub_control.publish(msg)
        self.get_logger().info('Command sent to drone.')

def main(args=None):
    rclpy.init(args=args)
    command_node = CommandNode()
    rclpy.spin(command_node)
    command_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

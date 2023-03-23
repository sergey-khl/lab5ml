#!/usr/bin/env python3

import rospy
import rospkg
import cv2
import pytorch as pt
from duckietown_msgs.srv import SetCustomLEDPattern, ChangePattern, ChangePatternResponse
from duckietown_msgs.msg import LEDPattern
from duckietown.dtros import DTROS, NodeType, TopicType
from std_msgs.msg import String

# References:   https://github.com/duckietown/dt-core/blob/6d8e99a5849737f86cab72b04fd2b449528226be/packages/led_emitter/src/led_emitter_node.py#L254
#               https://github.com/anna-ssi/mobile-robotics/blob/50d0b24eab13eb32d92fa83273a05564ca4dd8ef/assignment2/src/led_node.py

class DetectDigit(DTROS):
    def __init__(self, node_name: str) -> None:
        super(DetectDigit, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)

        # Zepeng Xiao from discord
        self.rospack = rospkg.RosPack()
        self.path = self.rospack.get_path("detect_digit") 
        self.trained_model = str(self.path)+"/src/tut1-model.pt"

        self.device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

        self.model = pt.jit.load('model_scripted.pt')
        self.model.eval()
        self.model = self.model.to(self.device)


        # -- Servers -- 
        self.server = rospy.Service(
            f'~led_pattern', 
            ChangePattern,
            self.findDatDigit
        )


    def findDatDigit(self, msg: ChangePattern):
        '''
        Changing the led msg to the one we want to use.
        '''
        
        print(msg)
        image = msg
        resized_image = cv2.resize(image, (28, 28))
        x = resized_image.to(self.device)

        y_pred, idk = self.model(x)
        print(y_pred, idk)
        

        return ChangePatternResponse()



if __name__ == "__main__":
    node = DetectDigit(node_name="detect_digit_node")
    rospy.spin()
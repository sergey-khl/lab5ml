#!/usr/bin/env python3

import rospy
import rospkg
import cv2
import torch as pt
import torch.nn as nn
import numpy as np
from duckietown.dtros import DTROS, NodeType, TopicType
from std_msgs.msg import String, Int32
from detect_digit.srv import img, imgResponse
from turbojpeg import TurboJPEG
import torch.nn.functional as F

# References:   https://github.com/duckietown/dt-core/blob/6d8e99a5849737f86cab72b04fd2b449528226be/packages/led_emitter/src/led_emitter_node.py#L254
#               https://github.com/anna-ssi/mobile-robotics/blob/50d0b24eab13eb32d92fa83273a05564ca4dd8ef/assignment2/src/led_node.py

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv2d(1,32,kernel_size =3, padding =1)
        self.batch = nn.BatchNorm2d(32)
        self.drop = nn.Dropout(0.4)
        self.input_fc = nn.Linear(25088, 2048)
        self.hidden_fc = nn.Linear(2048, 1024)
        self.output_fc = nn.Linear(1024, output_dim)

    def forward(self, x):

        # x = [batch size, height, width]

        batch_size = x.shape[0]
        

        #x = x.view(batch_size, -1)
        #print(x.shape)
        x = self.conv(x)
        
        x = self.batch(x)

        x = x.view(batch_size, -1)
        #print(x.shape)

        # x = [batch size, height * width]

        x = self.drop(x)

        h_1 = F.relu(self.input_fc(x))

        # h_1 = [batch size, 250]resized_image
        h_2 = F.relu(self.hidden_fc(h_1))
        h_2 = self.drop(h_2)

        # h_2 = [batch size, 100]

        y_pred = self.output_fc(h_2)

        # y_pred = [batch size, output dim]

        return y_pred, h_2

class DetectDigit(DTROS):
    def __init__(self, node_name: str) -> None:
        super(DetectDigit, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
            
        # Initialize TurboJPEG decoder
        self.jpeg = TurboJPEG()

        # Zepeng Xiao from discord
        self.rospack = rospkg.RosPack()
        self.path = self.rospack.get_path("detect_digit") 
        self.trained_model = str(self.path)+"/src/digitFinder.pt"

        self.device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

        self.model = MLP(0, 10)
        self.model.load_state_dict(pt.load(self.trained_model, map_location=pt.device('cpu')))
        self.model.eval()
        self.model = self.model.to(self.device)


        # -- Servers -- 
        self.server = rospy.Service(
            f'~detect_digit', 
            img,
            self.findDatDigit
        )


    def findDatDigit(self, req: img):
        '''
        Changing the led msg to the one we want to use.
        '''
        image = self.jpeg.decode(req.img.data)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(image, (28, 28))
        #cv2.imwrite(f'{str(self.path)}/src/data/test/{rospy.Time.now().secs}.jpg', resized_image)
        #cv2.imwrite(f'./test/{rospy.Time.now().secs}.jpg', resized_image)
        # if not cv2.imwrite(f'/data/test/{rospy.Time.now().secs}.jpg', resized_image):
        #     raise Exception("Could not write image")
        # x = np.float32(resized_image)
        # batch = pt.tensor(x / 255).unsqueeze(0).unsqueeze(0)

        # with pt.no_grad():
        #     batch = batch.to(self.device)
        #     # forward propagation
        #     y_pred, idk = self.model(batch)

        #     # get prediction
        #     output = pt.argmax(y_pred, 1).item()
        #     print("found number: ", output)
        

        return imgResponse(Int32(data=0))



if __name__ == "__main__":
    node = DetectDigit(node_name="detect_digit_node")
    rospy.spin()

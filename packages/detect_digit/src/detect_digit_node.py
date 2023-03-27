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
from sensor_msgs.msg import CameraInfo, CompressedImage
from turbojpeg import TurboJPEG
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.conv = nn.Conv2D()
        # self.input_fc = nn.Linear(input_dim, 2000)
        # self.hidden_fc = nn.Linear(2000, 1000)
        # self.output_fc = nn.Linear(1000, output_dim)
        #self.soft = nn.Softmax(dim=1)
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):

        # # x = [batch size, height, width]

        # batch_size = x.shape[0]

        # x = x.view(batch_size, -1)

        # # x = [batch size, height * width]

        # h_1 = F.relu(self.input_fc(x))

        # # h_1 = [batch size, 250]

        # h_2 = F.relu(self.hidden_fc(h_1))

        # # h_2 = [batch size, 100]

        # y_pred = F.log_softmax(self.output_fc(h_2), dim=1)


        # # y_pred = [batch size, output dim]

        # return y_pred, h_2
        x = self.conv1(x)
        x = self.conv2(x)        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

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

        self.model = MLP(784, 10)
        self.model.load_state_dict(pt.load(self.trained_model, map_location=pt.device('cpu')))
        self.model.eval()
        self.model = self.model.to(self.device)
        
        self.pub = rospy.Publisher("/robo/compressed",
                                   CompressedImage,
                                   queue_size=1)


        # -- Servers -- 
        self.server = rospy.Service(
            f'~detect_digit', 
            img,
            self.findDatDigit
        )


    def findDatDigit(self, req: img):
        image = self.jpeg.decode(req.img.data)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(image, (28, 28))
        
        x = np.float32(resized_image)
        batch = pt.tensor(x / 255).unsqueeze(0).unsqueeze(0)
        self.pub.publish(CompressedImage(format="jpeg", data=cv2.imencode('.jpg', resized_image)[1].tobytes()))
        with pt.no_grad():
            batch = batch.to(self.device)
            # forward propagation
            y_pred, _ = self.model(batch)

            # get prediction
            output = pt.argmax(y_pred, 1).item()
            if (y_pred[0][output] > 0.8):
                print(y_pred, output)
                return imgResponse(Int32(data=output))
            else:
                return imgResponse(Int32(data=-1))



if __name__ == "__main__":
    node = DetectDigit(node_name="detect_digit_node")
    rospy.spin()

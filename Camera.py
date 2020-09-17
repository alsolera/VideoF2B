##    VideoF2B v0.4 - Draw F2B figures from video
##    Copyright (C) 2018  Alberto Solera Rico - albertoavion(a)gmail.com
##    Copyright (C) 2020  Andrey Vasilik - basil96@users.noreply.github.com
##
##    This program is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation, either version 3 of the License, or
##    (at your option) any later version.
##
##    This program is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import cv2
#import tkinter
import tkinter.filedialog
import tkinter.simpledialog
from os import path

class CalCamera(object):
    def __init__(self, frame_size):
        
        try:
            CF = open('cal.conf', 'r')
            initialdir = CF.read()
            print(initialdir)
            CF.close()
        except:
            initialdir = '../'
        
        root = tkinter.Tk()
        root.withdraw() #use to hide tkinter window
        calibrationPath = tkinter.filedialog.askopenfilename(parent=root, initialdir=initialdir,
                                                       title='Select camera calibration .npz file or cancel to ignore')
        print(calibrationPath)
        #self.Calibrated = calibrationPath != ()
        self.Calibrated = len(calibrationPath)>1
        self.Located = False
        self.AR = True
        
        self.PointNames = ('circle center', 'front marker', 'left marker', 'right marker');

        self.frame_size = frame_size
        # Calibration default values
        self.map1 = None
        self.map2 = None

        if self.Calibrated:
            
            CF = open('cal.conf', 'w')
            CF.write(path.dirname(calibrationPath)) 
            CF.close()
    #       
            try:
                npzfile = np.load(calibrationPath)
                self.mtx = npzfile['mtx']
                self.dist = npzfile['dist']
                self.roi = npzfile['roi']
                self.newcameramtx = npzfile['newcameramtx']
                
                # Recalculate matrix and roi in case video from this camera was scaled after recording.
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, self.frame_size, 1)
                # For diagnostics only. If video was not scaled after recording, these comparisons should be exactly equal.
                # print(f'as recorded newcameramtx =\n{self.newcameramtx}')
                # print(f'scaled newcameramtx = \n{newcameramtx}')
                # print(f'as recorded roi =\n{self.roi}')
                # print(f'scaled roi =\n{roi}')
                self.newcameramtx = newcameramtx
                self.roi = roi

                # Calculate these undistortion maps just once
                self.map1, self.map2 = cv2.initUndistortRectifyMap(
                    self.mtx, self.dist, np.eye(3), self.newcameramtx, self.frame_size, cv2.CV_16SC2)

                self.cableLenght = tkinter.simpledialog.askfloat('Input', 'Total line lenght  (m) (Cancel = 21m)')
                if self.cableLenght is None: self.cableLenght = 21
                
                self.markRadius = tkinter.simpledialog.askfloat('Input', 'Height markers distance to center (m) (Cancel = 25m)')
                if self.markRadius is None: self.markRadius = 25
            except:
                print ('Error loading calibration file')
                self.Calibrated = False
                input("Press <ENTER> to continue without calibration...")
                
        CF.close()
        del root
        #del tkFileDialog
        #del tkSimpleDialog
        print('Using calibration: {}'.format(self.Calibrated))
    
    def Undistort (self, img):
        x, y, w, h = self.roi

        # img_slow = cv2.undistort(img.copy(), self.mtx, self.dist, None, self.newcameramtx)
        # # crop the image
        # img_slow = img_slow[y:y+h, x:x+w]

        # Faster method: calculate undistortion maps on init, then only call remap per frame.
        img = cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # crop it
        img = img[y:y+h, x:x+w]

        # Diagnostics for testing the equality of the faster approach against the original.
        # Uncomment above and below sections to verify.
        # np.save('img_undistort_slow', img_slow)
        # np.save('img_undistort_fast', img)
        # if not np.allclose(img_slow, img):
        #     raise ArithmeticError('img_slow and img are not exactly equal!')

        return img
    
    def Locate(self,img):
        self.calWindowName = 'Calibration (Center, Front, Left, Right)'
        
        rcos45 = self.markRadius * 0.70710678
        
        objectPoints = np.array([[0, 0, -1.5],
                         [0, self.markRadius, 0],
                         [-rcos45, rcos45, 0],
                         [rcos45, rcos45, 0]], dtype=np.float32)

        self.point = 0
        NumRefPoints = np.shape(objectPoints)[0]
        self.imagePoints = np.zeros((NumRefPoints, 2), dtype=np.float32)
        
        cv2.namedWindow(self.calWindowName, cv2.WINDOW_NORMAL )
        cv2.setMouseCallback(self.calWindowName,self.CB_mouse)
        
        while(1):
            
            cv2.imshow(self.calWindowName,
                       cv2.putText(img.copy(), 'Click ' + self.PointNames[self.point], (15, 20),  cv2.FONT_HERSHEY_TRIPLEX,.75, (0,0,255), 1))
            
            k = cv2.waitKey(1) & 0xFF
            
            if self.imagePoints[NumRefPoints-1,1] > 0:
                _ret, self.rvec, self.tvec = cv2.solvePnP(objectPoints, self.imagePoints,
                                                          self.newcameramtx, np.zeros_like(self.dist),
                                                          cv2.SOLVEPNP_ITERATIVE)
                self.Located = True
                cv2.destroyWindow(self.calWindowName)
                
                print(self.tvec)
                
                break
                
            if k == 27 or cv2.getWindowProperty(self.calWindowName, 1)<0:
                self.AR = False
                if cv2.getWindowProperty(self.calWindowName, 1)>=0:
                    cv2.destroyWindow(self.calWindowName)
                break
            
    
    # mouse callback function
    def CB_mouse(self,event,x,y,flags,param):
        global point,imagePoints
    
        if event == cv2.EVENT_LBUTTONDOWN:
            self.imagePoints[self.point,0],self.imagePoints[self.point,1] = x,y
            self.point += 1;
            #print self.imagePoints
            #cv2.circle(self.calWindowName,(x,y),1,(0,255,0),-1)
    
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.point >0:
                self.point -= 1;
#                 cv2.circle(self.calWindowName,
#                            (imagePoints[point,0],imagePoints[point,1])
#                            ,1,(255,255,255),-1)
                self.imagePoints[self.point,0] = 0
                self.imagePoints[self.point,1] = 0
                #print self.imagePoints
        


import cv2
import numpy as np

def read_image(filename):
    im=cv2.imread(filename)
    return im

def edge_detection(image,line_width,blur_amount):
    gray_scale_img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    grayBlur=cv2.medianBlur(gray_scale_img,blur_amount)
    edges=cv2.adaptiveThreshold(grayBlur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,line_width,blur_amount)
    return edges

def color_quantisation(image,k):
    data=np.float32(image).reshape((-1,3))
    criteria=(cv2.TermCriteria_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,0.001)
    ret,label,centroid=cv2.kmeans(data,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    centroid=np.uint8(centroid)
    result=centroid[label.flatten()]
    result=result.reshape(image.shape)
    return result

image=read_image('./bawaal.jpg')
line_width=9
blur_amount=7
totalColors=6
edgeimage=edge_detection(image,line_width,blur_amount)
image=color_quantisation(image,totalColors)
blurred_image=cv2.bilateralFilter(image,d=7,sigmaColor=200,sigmaSpace=200)
cartoon=cv2.bitwise_and(blurred_image,blurred_image,mask=edgeimage)
cv2.imwrite('cartoon.jpg',cartoon)
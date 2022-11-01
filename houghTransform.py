import cv2

import numpy as np
import matplotlib.pyplot as plt

import math


def frange(start, stop=None, step=None):
    # if set start=0.0 and step = 1.0 if not specified
    start = float(start)
    if stop == None:
        stop = start + 0.0
        start = 0.0
    if step == None:
        step = 1.0

    count = 0
    while True:
        temp = float(start + count * step)
        if step > 0 and temp >= stop:
            break
        elif step < 0 and temp <= stop:
            break
        yield temp
        count += 1

def line_detection(image, edge_image, num_rhos=180, num_thetas=180, t_count=500):
  edge_height, edge_width = edge_image.shape[:2]
  edge_height_half, edge_width_half = edge_height/2 , edge_width/2
  
  d = math.sqrt(edge_height**2 + edge_width**2)
  dtheta = 180 / num_thetas
  drho = (2 * d) / num_rhos
  
  
  thetas = list(range(0,180,int(dtheta)))
  
  rhos= list(frange(-d,d,drho))


 
  cos_thetas = []
  sin_thetas = []
  for j in range(len(thetas)):
    cos_thetas.append(math.cos(math.radians(thetas[j])))
    sin_thetas.append(math.sin(math.radians(thetas[j])))
  #print(cos_thetas)
  

  #2D Array accumulator representing the Hough Space with dimension (len(rhos), len(thetas))
  accumulator = [ [0]*len(rhos) for _ in range(len(thetas)) ]
  
  
  #Gráfica de Hough
  figure = plt.figure(figsize=(12, 12))
  subplot3 = figure.add_subplot(1, 1, 1)
  subplot3.set_facecolor((0, 0, 0))

  rabs_list = []
 
  
  #
  for y in range(edge_height):
    for x in range(edge_width):
      if edge_image[y][x] != 0:
        edge_point = [y - edge_height_half, x - edge_width_half]
        ys, xs = [], []
        for theta_idx in range(len(thetas)):
          rho = (edge_point[1] * cos_thetas[theta_idx]) + (edge_point[0] * sin_thetas[theta_idx])
          theta = thetas[theta_idx]
          for r in range(len(rhos)):
            rabs = abs(rhos[r]-rho)
            rabs_list.append(rabs)
            rabs_list2 = rabs_list 
            if(len(rabs_list)==num_rhos):
              rabs_list=[]

          min_value = min(rabs_list2)
          rho_idx = rabs_list2.index(min_value)
          #rho_idx = np.argmin(np.abs(rhos - rho))
          
          #print(rho_idx)
          accumulator[rho_idx][theta_idx] += 1
          ys.append(rho)
          xs.append(theta)
        subplot3.plot(xs, ys, color="blue", alpha=0.05)
  print(accumulator)
  for y in range(len(accumulator)):
    for x in range(len(accumulator)):
      if accumulator[y][x] > t_count:
        rho = rhos[y]
        theta = thetas[x]
        a = math.cos(math.radians(theta))
        b = math.sin(math.radians(theta))
        x0 = (a * rho) + edge_width_half
        y0 = (b * rho) + edge_height_half
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        subplot3.plot([theta], [rho], marker='o', color="red")
        cv2.line(image,(x1,y1),(x2,y2),(0, 0, 255),2)
        cv2.imshow("Lineas detectadas " + str(i),image)
        cv2.waitKey(0)

  subplot3.invert_yaxis()
  subplot3.invert_xaxis()


  subplot3.title.set_text("Transformada de Hough")
  
  plt.show()
  return accumulator, rhos, thetas


if __name__ == "__main__":
  for i in range(7):
    image = cv2.imread(f"sample-{i+1}.png")
    edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.GaussianBlur(edge_image, (3, 3), 1)
    edge_image = cv2.Canny(edge_image, 100, 200)
    edge_image = cv2.dilate(
        edge_image,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )
    edge_image = cv2.erode(
        edge_image,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )
    cv2.imshow("Imagen original "+str(i),image)
    cv2.waitKey(0)

    cv2.imshow("Imagen de bordes "+str(i),edge_image)
    cv2.waitKey(0)

    #Line detection propio
    line_detection(image, edge_image)
    #Line Detection opencv
    lines = cv2.HoughLines(edge_image,1,np.pi/180,200)
    for line in lines:
      rho,theta = line[0]
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + 1000*(-b))
      y1 = int(y0 + 1000*(a))
      x2 = int(x0 - 1000*(-b))
      y2 = int(y0 - 1000*(a))
      cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow("Hough OpenCV" + str(i),image)

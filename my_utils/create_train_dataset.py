import json
import pandas as pd
import os
import shutil
def read_and_join(filename):
    f = open(filename)
    data=json.load(f)
    images=pd.DataFrame(data["images"])
    annotations=pd.DataFrame(data["annotations"])
    categories=pd.DataFrame(data["categories"])
    print(f"images: {len(images)} \nannotations: {len(annotations)} \ncategories:{len(categories)}")
    print("---------------------------------------------------------------------------------------")
    img_ann=pd.merge(images.rename(columns={"id":"image_id"}),annotations,on="image_id",how="inner")
    img_ann_cat=pd.merge(categories.rename(columns={"id":"category_id"}),img_ann,on="category_id",how="inner")
    print(f"img_ann: {len(img_ann)} \nimg_ann_cat: {len(img_ann_cat)}")
    print("---------------------------------------------------------------------------------------")
    return img_ann_cat
def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]
def save_labels(file_name,row,labels_path,file_name_w_o_ext):

    cat_col="category_id"
    category_ids=row[cat_col]
    bbox=row["bbox"]
    width=list(row["width"])[0]
    height=list(row["height"])[0]
    newbox=[]
    for eachbox in list(bbox):
        newbox+=[coco_to_yolo(eachbox[0],eachbox[1],eachbox[2],eachbox[3],width,height)]


    boxes_df=pd.DataFrame(newbox)
    new_col=["x",'y',"w","h"]
    boxes_df.columns=new_col
    boxes_df[cat_col]=list(category_ids)
    filter_data=boxes_df[[cat_col,*new_col]]
    com_labels_path=os.path.join(labels_path,file_name_w_o_ext+".txt")
    filter_data.to_csv(com_labels_path,sep=" ",header=False,index=False)




def create_yolo_dataset(data,root_path,from_path):
    image_path=os.path.join(root_path,"images")
    labels_path=os.path.join(root_path,"labels")
    train=pd.DataFrame(columns=["image","label"])
    data_g=data.groupby("file_name")
    for file_name, row in data_g:
        img_w_ext=file_name
        file_name_w_o_ext = file_name[:len(file_name) - 5]
        save_labels(file_name,row,labels_path,file_name_w_o_ext)
        dict={"image":img_w_ext,"label":file_name_w_o_ext+".txt"}
        dict_df=pd.DataFrame(dict,index=[0])
        train=pd.concat([train,dict_df])
        com_img_path_src=os.path.join(from_path,img_w_ext)
        shutil.copy(com_img_path_src,image_path)
    return train


datacsv=read_and_join("../../train_gt.json")
sample=datacsv.head(1000)
root_path= "../../dataset"
traincsv=create_yolo_dataset(sample, root_path, "../../dataset/train_images")
print(f"traincsv: {len(traincsv)}")
traincsv.to_csv(root_path+"/train.csv",index=False)
print("..Done")






# from yolov2.train import  main
# import gc
# gc.collect()
# if __name__ == '__main__':
#     main()
import sys

import matplotlib.pyplot as plt
import json
import pandas as pd
import os

import numpy as np
from PIL import Image
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil






def read_and_join(filename):
    f = open(filename)
    data = json.load(f)
    images = pd.DataFrame(data["images"])
    annotations = pd.DataFrame(data["annotations"])
    categories = pd.DataFrame(data["categories"])
    print(f"images: {len(images)} \nannotations: {len(annotations)} \ncategories:{len(categories)}")
    print("---------------------------------------------------------------------------------------")
    img_ann = pd.merge(images.rename(columns={"id": "image_id"}), annotations, on="image_id", how="inner")
    img_ann_cat = pd.merge(categories.rename(columns={"id": "category_id"}), img_ann, on="category_id", how="outer")
    print(f"img_ann: {len(img_ann)} \nimg_ann_cat: {len(img_ann_cat)}")
    print("---------------------------------------------------------------------------------------")
    return img_ann_cat


def filter(df):
    newdata = pd.DataFrame()

    #________________StopSign
    StopSign = datacsv[datacsv["name"] == "StopSign"]
    newdata = pd.concat([newdata, StopSign])
    len_of_stop_Sign= len(StopSign)
    print(f"stopSign: {len(StopSign)}    newdata:  {len(newdata)} ")


    # ________________traffic_lights
    traffic_lights = df[df["name"] == "traffic_lights"]
    temp =traffic_lights.sort_values("file_name")
    traffic_lights = temp[0: len_of_stop_Sign]
    if (len(traffic_lights) !=len_of_stop_Sign):
        rem = len_of_stop_Sign - len(traffic_lights)
        traffic_lights=pd.concat([traffic_lights,temp[0: rem]])
    newdata = pd.concat([newdata, traffic_lights])
    print(f"traffic_lights: {len(traffic_lights)}   newdata:  {len(newdata)}")




    # ________________Car
    Car = df[df["name"] == "Car"]
    temp = Car.sort_values("file_name")
    Car = temp[0: len_of_stop_Sign]
    if (len(Car) != len_of_stop_Sign):
        rem = len_of_stop_Sign - len(Car)
        Car = pd.concat([Car, temp[0: rem]])
    newdata = pd.concat([newdata, Car])
    print(f"Car: {len(Car)}   newdata:  {len(newdata)}")

    # ________________Truck
    Truck = df[df["name"] == "Truck"]
    temp = Truck.sort_values("file_name")
    Truck = temp[0: len_of_stop_Sign]
    if (len(Truck) != len_of_stop_Sign):
        rem = len_of_stop_Sign - len(Truck)
        Truck = pd.concat([Truck, temp[0: rem]])
    newdata = pd.concat([newdata, Truck])
    print(f"Truck: {len(Truck)}   newdata:  {len(newdata)}")
    return newdata.sort_values("file_name")
def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    x_center = x1 + w / 2
    y_center = y1 + h / 2
    x_center /= image_w
    y_center /= image_h
    w /= image_w
    h /= image_h
    #return [((2 * x1 + w) / (2 * image_w)), ((2 * y1 + h) / (2 * image_h)), w / image_w, h / image_h]
    return [x_center,y_center,w,h]

def save_labels(file_name, row, labels_path, file_name_w_o_ext, com_img_path_src, image_path):
    cat_col = "category_id"
    category_ids = row[cat_col]
    bbox = row["bbox"]
    width = list(row["width"])[0]
    height = list(row["height"])[0]

    newbox = []
    bbox_l = list(bbox)
    labels = list(row["category_id"])
    names = list(list(row["name"]))
    try:
        # image = Chitra(com_img_path_src, bbox_l,names,)
        # print(image.shape)

        # Chitra can rescale your bounding box automatically based on the new image size.
        # a=image.resize_image_with_bbox((720, 720))
        # print(image.shape)
        # bbox_l = image.bboxes

        # img = np.array(image.image)#cv2.imread(com_img_path_src)
        bbox = list(bbox)
        for i, eachbox in enumerate(bbox):
            eachanme = names[i]
            # x = int(np.ceil(bbox_l[i].x2))
            # y = int(np.ceil(bbox_l[i].y2))
            # w = int(np.ceil(bbox_l[i].x1))
            # h = int(np.ceil(bbox_l[i].x1))

            x, y, w, h = int(float(eachbox[0])), int(float(eachbox[1])), int(float(eachbox[2])), int(float(eachbox[3]))
            newbox += [coco_to_yolo(x, y, w, h, width, height)]
            # newbox += [[x, y, w, h]]
            # x,y,w,h=bbox_l[0],bbox_l[1],bbox_l[2],bbox_l[3]

            # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # img = cv2.putText(img, eachanme, (x + w + 10, y + h), 0, 0.3, (0, 255, 0))
        # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imshow(com_img_path_src[10:], img)
        #
        # cv2.waitKey(0)
        #
        # cv2.destroyAllWindows()

        boxes_df = pd.DataFrame(newbox)
        new_col = ["x", 'y', "w", "h"]
        boxes_df.columns = new_col
        boxes_df[cat_col] = list(category_ids)
        filter_data = boxes_df[[cat_col, *new_col]]
        com_labels_path = os.path.join(labels_path, file_name_w_o_ext + ".txt")

    except Exception as e:
        print(e)
        sys.exit()
        return False
    filter_data.to_csv(com_labels_path, sep=" ", header=False, index=False)
    # cv2.imwrite(os.path.join(image_path,file_name), img)
    return True


def create_yolo_dataset(data, root_path, from_path):
    image_path = os.path.join(root_path, "images")
    labels_path = os.path.join(root_path, "labels")
    train = pd.DataFrame(columns=["image", "label"])
    data_g = data.groupby("file_name")

    for file_name, row in data_g:
        img_w_ext = file_name
        file_name_w_o_ext = file_name[:len(file_name) - 5]
        com_img_path_src = os.path.join(from_path, img_w_ext)
        flag = save_labels(file_name, row, labels_path, file_name_w_o_ext, com_img_path_src, image_path)
        if flag:
            dict = {"image": img_w_ext, "label": file_name_w_o_ext + ".txt"}
            dict_df = pd.DataFrame(dict, index=[0])
            train = pd.concat([train, dict_df])

            # save_labels(file_name, row, labels_path, file_name_w_o_ext,com_img_path_src,image_path)
            shutil.copy(com_img_path_src, image_path)
    return train


datacsv = read_and_join("dataset/train_gt.json")



# #---------------temp
groups=datacsv.groupby(["file_name"])

completed=groups[['category_id', 'name', 'supercategory', 'image_id','height', 'width', 'bbox', 'id']].agg(list).reset_index()
completed_sort=completed.sort_values("name")
completed_names=completed.sort_values("name")

#1-4
StopSign=completed[completed["name"].map(lambda x:len({"StopSign"}.symmetric_difference(set(x)))==0)]
traffic_lights=completed[completed["name"].map(lambda x:len({"traffic_lights"}.symmetric_difference(set(x)))==0)]
Car=completed[completed["name"].map(lambda x:len({"Car"}.symmetric_difference(set(x)))==0)]
Truck=completed[completed["name"].map(lambda x:len({"Truck"}.symmetric_difference(set(x)))==0)]

#5 StopSign Car
StopSign_Car=completed[completed["name"].map(lambda x:len({"Car","StopSign"}.symmetric_difference(set(x)))==0)]


#6 StopSign_Truck
StopSign_Truck=completed[completed["name"].map(lambda x:len({"Truck","StopSign"}.symmetric_difference(set(x)))==0)]

#7 StopSign_traffic_lights
StopSign_traffic_lights=completed[completed["name"].map(lambda x:len({"traffic_lights","StopSign"}.symmetric_difference(set(x)))==0)]


#8 StopSign_Car_Truck
StopSign_Car_Truck=completed[completed["name"].map(lambda x:len({"Truck","Car","StopSign"}.symmetric_difference(set(x)))==0)]

#9 StopSign_Car_traffic_lights
StopSign_Car_traffic_lights=completed[completed["name"].map(lambda x:len({"traffic_lights","Car","StopSign"}.symmetric_difference(set(x)))==0)]
#10 StopSign_Truck_traffic_lights
StopSign_Truck_traffic_lights=completed[completed["name"].map(lambda x:len({"traffic_lights","Truck","StopSign"}.symmetric_difference(set(x)))==0)]

#11 Car_Truck
Car_Truck=completed[completed["name"].map(lambda x:len({"Truck","Car"}.symmetric_difference(set(x)))==0)]
#12 Car_traffic_lights
Car_traffic_lights=completed[completed["name"].map(lambda x:len({"traffic_lights","Car"}.symmetric_difference(set(x)))==0)]
#13 Truck_traffic_lights
Truck_traffic_lights=completed[completed["name"].map(lambda x:len({"traffic_lights","Truck"}.symmetric_difference(set(x)))==0)]
#14alll
Car_Truck_traffic_lights_StopSign=completed[completed["name"].map(lambda x:set(["traffic_lights","Truck","Car","StopSign"]).issubset(set(x)))]




biased_batch=pd.concat([StopSign_Truck_traffic_lights,
                        Truck_traffic_lights,
                        StopSign_traffic_lights,
                        Car_Truck_traffic_lights_StopSign,
                        traffic_lights,StopSign_Truck,
                        StopSign_Car_traffic_lights,
                        StopSign, StopSign_Car_Truck
                        ])

Car_traffic_lights =Car_traffic_lights.sort_values("file_name")
StopSign_Car =StopSign_Car.sort_values("file_name")
Truck =Truck.sort_values("file_name")
Car =Car.sort_values("file_name")
Car_Truck =Car_Truck.sort_values("file_name")

count=1
Regular_batch1=pd.concat([
    Car_traffic_lights[242*(count-1):242*count],
    StopSign_Car[313*(count-1):313*count],
    Truck[433*(count-1):433*count],
    Car[2472*(count-1):2472*count],
    Car_Truck[2565*(count-1):2565*count],
    ])



count=2
Regular_batch2=pd.concat([
    Car_traffic_lights[242*(count-1):242*count],
    StopSign_Car[313*(count-1):313*count],
    Truck[433*(count-1):433*count],
    Car[2472*(count-1):2472*count],
    Car_Truck[2565*(count-1):2565*count],
    ])


count=3
Regular_batch3=pd.concat([
    Car_traffic_lights[242*(count-1):242*count],
    StopSign_Car[313*(count-1):313*count],
    Truck[433*(count-1):433*count],
    Car[2472*(count-1):2472*count],
    Car_Truck[2565*(count-1):2565*count],
    ])




count=4
Regular_batch4=pd.concat([
    Car_traffic_lights[242*(count-1):242*count],
    StopSign_Car[313*(count-1):313*count],
    Truck[433*(count-1):433*count],
    Car[2472*(count-1):2472*count],
    Car_Truck[2565*(count-1):2565*count],
    ])




count=5
Regular_batch5=pd.concat([
    Car_traffic_lights[242*(count-1):242*count],
    StopSign_Car[313*(count-1):313*count],
    Truck[433*(count-1):433*count],
    Car[2472*(count-1):2472*count],
    Car_Truck[2565*(count-1):2565*count],
    ])




count=6
Regular_batch6=pd.concat([
    Car_traffic_lights[242*(count-1):242*count],
    StopSign_Car[313*(count-1):313*count],
    Truck[433*(count-1):433*count],
    Car[2472*(count-1):2472*count],
    Car_Truck[2565*(count-1):2565*count], n
    ])



set1=pd.concat([Regular_batch1,biased_batch])
set2=pd.concat([Regular_batch2,biased_batch])
set3=pd.concat([Regular_batch3,biased_batch])
set4=pd.concat([Regular_batch4,biased_batch])
set5=pd.concat([Regular_batch5,biased_batch])
set6=pd.concat([Regular_batch6,biased_batch])







#---------------------temp

groups=datacsv.group("file_name")



# sample=datacsv.head(0000)
# datacsv_g=datacsv.groupby("name")
# Truck=datacsv_g.get_group("Truck").head(1000)
# Car=datacsv_g.get_group("Car").head(1000)
# StopSign=datacsv_g.get_group("StopSign").head(1000)
# traffic_lights=datacsv_g.get_group("traffic_lights").head(1000)
# datacsv_g_concat=pd.concat([Truck,traffic_lights,StopSign,Car])
root_path = "yolov1/data/set1"

datacsv = filter(datacsv)
print("sending to convert", len(datacsv))
traincsv = create_yolo_dataset(datacsv.head(5000) ,root_path, "dataset/train_images")
print(f"traincsv: {len(traincsv)}")

traincsv.to_csv(root_path + "/train.csv", index=False)
print("..Done")

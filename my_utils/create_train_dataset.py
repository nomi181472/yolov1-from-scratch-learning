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

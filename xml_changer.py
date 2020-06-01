import xml.etree.ElementTree as ET
import numpy as np
import skimage.transform
import xml
from PIL import Image
import cv2
from random import uniform
import shutil
import glob
import os

def read_xml_regression_and_name(xml_path):
    e = xml.etree.ElementTree.parse(xml_path).getroot()
    ob = e.findall('object')
    # print(ob)

    output_list = [[] for x in range(2)]

    for atype in e.findall('object'):
        name = atype.find('name').text
        output_list[0].append(name)
        bbox = atype.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        regression_list = [xmin,ymin,xmax,ymax]
        output_list[1].append(regression_list)

    return output_list[0],output_list[1]

def xml_bbox_location_changer(dataset_train,xml_path,scale,change_name=None):
    def parse_rec(filename):
        """ Parse a PASCAL VOC xml file """
        filename = os.path.join(dataset_train, filename)
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
            objects.append(obj_struct)

        return objects

    def update_xml(filename, objects):
        """ Parse a PASCAL VOC xml file """
        xml_file = os.path.join(dataset_train, filename)
        tree = ET.parse(xml_file)

        objects_an = {obj['name']: obj['bbox'] for obj in objects}
        # print('an', objects_an)
        root = tree.getroot()
        obj_xml = root.findall('object')
        for obj in obj_xml:
            name = obj.find('name')
            bbox_mod = objects_an[name.text]  # do a try catch here
            # print('mod',bbox_mod)

            bbox_original = obj.find('bndbox')
            bbox_original.find('xmin').text = str(bbox_mod[0])
            bbox_original.find('ymin').text = str(bbox_mod[1])
            bbox_original.find('xmax').text = str(bbox_mod[2])
            bbox_original.find('ymax').text = str(bbox_mod[3])

            # xml label 이름을 변경하고 싶으면 해당 변수를 변경
            if change_name != None:
                obj.find('name').text = change_name

        tree.write(xml_file)

    print(dataset_train)
    file_in_sub_folder = glob.glob(os.path.join(dataset_train, "*.png"))
    for ind_file in range(len(file_in_sub_folder)):
        img = cv2.imread(file_in_sub_folder[ind_file], 0)
        print(file_in_sub_folder[ind_file])

        xml_file_name = file_in_sub_folder[ind_file][file_in_sub_folder[ind_file].rindex('/') + 1:]
        xml_file = xml_file_name.replace('.png','.xml')
        xml_file = xml_path + '/' + xml_file
        print(xml_file)
        # xml_file = os.path.splitext(os.path.basename(file_in_sub_folder[ind_file]))[0] + '.xml'
        recs = parse_rec(xml_file)
        target_class = ['negative']
        objects = []
        for index_cls in range(len(target_class)):
            R = [obj for obj in recs if obj['name'] == target_class[index_cls]]
            print(R)
            # for idx in range(len(R)):
            bbox_gt = np.array([x['bbox'] for x in R])
            bbox_gt = bbox_gt[0]
            print('bbox_gt',bbox_gt)

            xmin = bbox_gt[0]
            xmax = bbox_gt[2]
            ymin = bbox_gt[1]
            ymax = bbox_gt[3]

            print('img shape ',img.shape)
            if xmin < 0:
                xmin = 0
            if xmax > img.shape[1]:
                xmax = img.shape[1]
            if ymin < 0:
                ymin = 0
            if ymax > img.shape[0]:
                ymax = img.shape[0]

            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
            print(bbox)

            obj_struct = {}
            obj_struct['name'] = target_class[index_cls]
            obj_struct['bbox'] = bbox
            objects.append(obj_struct)

            print(objects)

        update_xml(xml_file, objects)

def use_open_change_xml_name():
    xml_path = '/media/j/DATA/LabelImage/fracture_label/wrist/professor_fix_label/add_xml/4_roi_wrist_detection_train/retinanet_train/part_detection_la/val/xml' + '/'
    save_path = '/media/j/DATA/LabelImage/fracture_label/wrist/professor_fix_label/add_xml/4_roi_wrist_detection_train/retinanet_train/part_detection_la/val/xml_save' + '/'
    xml_list = os.listdir(xml_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for xml_file in xml_list:
        with open(xml_path + xml_file, 'r') as f:
            content = f.readlines()
            new_content = []
            for line in content:
                if 'negative' in line:
                    fix = line.replace('_negative','')
                    new_content.append(fix)
                elif 'frac' in line:
                    fix = line.replace('_frac','')
                    new_content.append(fix)
                else:
                    new_content.append(line)

        with open(save_path + xml_file, 'w') as fw:
            fw.writelines(new_content)

def use_open_change_xml_location():
    xml_path = '/media/j/DATA/fracture/segmentation_frac/radius_set/crop_sr_set/val_xml' + '/'
    save_path = '/media/j/DATA/fracture/segmentation_frac/radius_set/crop_sr_set/val_xml_fix' + '/'
    xml_list = os.listdir(xml_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    def fix_location(line):
        left_idx = line.index('>')
        right_idx = line.rindex('<')

        location = int(line[left_idx + 1:right_idx])
        fix_location = str(location * 4)
        left = line[:left_idx + 1]
        right = line[right_idx:]
        fix = left + fix_location + right
        print('line', line)
        print('fix', fix)

        return fix

    for xml_file in xml_list:
        if 'patient' in xml_file:
            with open(xml_path + xml_file, 'r') as f:
                content = f.readlines()
                new_content = []
                for line in content:
                    if 'xmin' in line:
                        fix = fix_location(line)
                        new_content.append(fix)
                    elif 'ymin' in line:
                        fix = fix_location(line)
                        new_content.append(fix)
                    elif 'xmax' in line:
                        fix = fix_location(line)
                        new_content.append(fix)
                    elif 'ymax' in line:
                        fix = fix_location(line)
                        new_content.append(fix)
                    else:
                        new_content.append(line)

            with open(save_path + xml_file, 'w') as fw:
                fw.writelines(new_content)



if __name__ == '__main__':
    # img_path = '/media/j/B50F-BCF1/as/scaphoid/HandBone/sca_nega'
    # xml_path = '/media/j/B50F-BCF1/as/scaphoid/roiloc/sca_nega'
    # xml_bbox_location_changer(img_path,xml_path,1,'scaphoid_negative')
    # print(read_xml_regression_and_name(xml_path))
    # use_open_change_xml_name()
    use_open_change_xml_location()
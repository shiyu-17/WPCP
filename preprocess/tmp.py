import json

def add_pose_to_annotations(json_data):
    # 将images列表转换为以image_id为键的字典
    image_dict = {image['id']: image['pose'] for image in json_data['images']}
    
    # 遍历annotations，根据image_id添加相同的pose
    for annotation in json_data['annotations']:
        image_id = annotation['image_id']
        if image_id in image_dict:
            print(annotation['image_id'])
            annotation['pose'] = image_dict[image_id]
    
    return json_data

# 读取JSON文件
with open('E:\dataset\pic_960\keypoints.json', 'r') as file:
    json_data = json.load(file)

# 调用函数并保存结果
new_json_data = add_pose_to_annotations(json_data)

# 将结果写入新的JSON文件
with open(r'E:\dataset\pic_960\new_json_file.json', 'w') as file:
    json.dump(new_json_data, file, indent=4)

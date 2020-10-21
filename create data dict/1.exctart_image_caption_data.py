import json
import pickle

val_path='.\\annotations\\captions_val2014.json'
train_path='.\\annotations\\captions_train2014.json'

with open(val_path) as json_data:
    file = json.load(json_data)
data_str = json.dumps(file)
val_dict= json.loads(data_str)
with open(train_path) as json_data:
    file = json.load(json_data)
data_str = json.dumps(file)
train_dict= json.loads(data_str)
   
print(train_dict.keys())
print(val_dict.keys())
print(train_dict['annotations'][0]['caption'])
print(val_dict['annotations'][0].keys())
print('------------------train------------------')
print('count_img')
print(len(train_dict['images']))
print('count_captions')
print(len(train_dict['annotations']))
print('------------------val------------------')
print('count_img')
print(len(val_dict['images']))
print('count_captions')
print(len(val_dict['annotations']))
max_width=0
max_height=0
dict_id=dict()
dict_id['train']=[]
dict_id['val']=[]

dict_data=dict()
dict_data['train']=dict()
dict_data['val']=dict()
print("******************* images data extraction *******************")
######################## extract images #######################################
for image in train_dict['images']:
    dict_id['train'].append(image['id'])
    dict_data['train'][image['id']]={'width':image['width'],'height':image['height'],'url':image['coco_url']}
    
for image in val_dict['images']:
    dict_id['val'].append(image['id'])
    if max_width <image['width']:
        max_width =image['width']
    if max_height <image['height']:
        max_height =image['height']
    dict_data['val'][image['id']]={'width':image['width'],'height':image['height'],'url':image['coco_url']}

######################## extract captions #####################################
print("******************* captions data extraction *******************")
for annotation in train_dict['annotations']:
    if 'captions' in dict_data['train'][annotation['image_id']].keys():
        dict_data['train'][annotation['image_id']]['captions'].append(annotation['caption'])
    else:
        dict_data['train'][annotation['image_id']]['captions']=[annotation['caption']]

for annotation in val_dict['annotations']:
    if 'captions' in dict_data['val'][annotation['image_id']].keys():
        dict_data['val'][annotation['image_id']]['captions'].append(annotation['caption'])
    else:
        dict_data['val'][annotation['image_id']]['captions']=[annotation['caption']]

print(max_width,max_height)
print('------------------train------------------')
print('count_img',len(dict_data['train']))
print('count_captions',sum([len(img['captions']) for id,img in dict_data['train'].items() ]))
print('count_img with more then 5 captions')
print(len([len(img['captions']) for id,img in dict_data['train'].items() if len(img['captions'])!=5 ]))

print('------------------val------------------')
print('count_img',len(dict_data['val']))
print('count_captions',sum([len(img['captions']) for id,img in dict_data['val'].items() ]))
print('count_img with more then 5 captions')
print(len([len(img['captions']) for id,img in dict_data['train'].items() if len(img['captions'])!=5  ]))
####################### save data as pickle ####################################
pickle_file=open("dataset.pickle","wb")
pickle.dump([dict_id,dict_data],pickle_file)
pickle_file.close()

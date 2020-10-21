import json, pickle, gc

print("************** Extracting train annotations data **************\n")
train_path='.\\annotations\\instances_train2014.json'
with open(train_path) as json_data:
    file = json.load(json_data)
data_str = json.dumps(file)
del file
gc.collect()
train_dict= json.loads(data_str)
del data_str
gc.collect()
file = open('dataset.pickle', 'rb')
[dict_id,dict_data]=pickle.load(file)
file.close()

for annotation in train_dict['annotations']:
    anno_info={'id':annotation['id'],'iscrowd':annotation['iscrowd'],'category_id':annotation['category_id'],'bbox':annotation['bbox']}
    if 'annotations' in dict_data['train'][annotation['image_id']].keys():
        dict_data['train'][annotation['image_id']]['annotations'].append(anno_info)
    else:
        dict_data['train'][annotation['image_id']]['annotations']=[anno_info]
cnt=0
error=0
for img in dict_data['train'].values():
    try:
        cnt+=len(img['annotations'])
    except:
        error+=1   
print('count annotations:',len( train_dict['annotations']))
print('count images:',len( train_dict['images']))
print('count images without annotation :',error,'=',100.*error/len( train_dict['images']),'%')
del train_dict
gc.collect()

print("************** Extracting val annotations data **************\n")
val_path='.\\annotations\\instances_val2014.json'
with open(val_path) as json_data:
    file = json.load(json_data)
data_str = json.dumps(file)
del file
gc.collect()
val_dict= json.loads(data_str)
del data_str
gc.collect()

categories=[]
for categorie in val_dict['categories']:
    categories.append(categorie)
pickle_file=open("categories.pickle","wb")
pickle.dump(categories,pickle_file)
pickle_file.close()
for annotation in val_dict['annotations']:
    anno_info={'id':annotation['id'],'iscrowd':annotation['iscrowd'],'category_id':annotation['category_id'],'bbox':annotation['bbox']}
    if 'annotations' in dict_data['val'][annotation['image_id']].keys():
        dict_data['val'][annotation['image_id']]['annotations'].append(anno_info)
    else:
        dict_data['val'][annotation['image_id']]['annotations']=[anno_info]
cnt=0
error=0
for img in dict_data['val'].values():
    try:
        cnt+=len(img['annotations'])
    except:
        error+=1   
print('count annotations:',len( val_dict['annotations']))
print('count images:',len( val_dict['images']))
print('count images without annotation :',error,'=',100.*error/len( val_dict['images']),'%')

####################### save data as pickle ####################################
pickle_file=open("dataset.pickle","wb+")
pickle.dump([dict_id,dict_data],pickle_file)
pickle_file.close()



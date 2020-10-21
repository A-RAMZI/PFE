
import tensorflow as tf 
import numpy as np 
import pickle
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pickle
import pathlib
import urllib.request
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from PyQt5 import QtCore, QtGui, QtWidgets
from os import path
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
    
########################## functions ##########################
def load_data(str='dataset.pickle'):
    file = open(str, 'rb')
    [dict_id,dict_data]=pickle.load(file)
    file.close()
    print(len(dict_id['train']))
    print(len(dict_id['val']))
    return [dict_id,dict_data]
    
def get_page(data_type,page,dict_id):
    first_index=page*10
    last_index=min((page+1)*10,len(dict_id[data_type]))

    list_data=[]
    for i in range(first_index,last_index):
        list_data.append(' %05d'%i+' :'+str(dict_id[data_type][i]))
    return list_data

def isValidPage(data_type,dict_id,page):
    max_pages=int((len(dict_id[data_type])+9.0)/10)
    print(max_pages)
    if page=='' :
        page='0'
        return 0
    try:
        numPage=int(page)
    except:   
        numPage=-1
    if (0<=numPage) and (numPage<max_pages):
        return numPage
    return -1

def getSelectedType(index):
    list=['train','val']
    return list[index]




def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))

  return model

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def get_features(model, image_path):
    image_np = np.array(Image.open(image_path))
    output_dict = run_inference_for_single_image(model, image_np)
    vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'] ,
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks_reframed', None),
    use_normalized_coordinates=True,
    line_thickness=8)
    features=[]
    features.extend( output_dict['detection_boxes'][0:30].reshape(-1))
    features.extend(output_dict['detection_classes'][0:30])
    features.extend(output_dict['detection_scores'][0:30])
    return (image_np,features)


def predict_sent( model, features):

	char_list = pickle.load(open('char_list.pickle', 'rb'))
	sent=""
	last_output=char_list['begin'] 
	keys_list=[key for key in char_list]
	i=0
	input=[]
	input.extend(features)
	input.append(last_output)
	model.reset_states()
	data=np.zeros((64,100,181))
	data[0,i]=np.array(input)
	output=model(data)
	last_output=np.argmax(output[0,i])
	while last_output != char_list['None'] and i<99:
		sent+=keys_list[last_output]
		input=[]
		input.extend(features)
		input.append(last_output)
		i+=1
		data[0,i]=np.array(input)
		output=model(data)
		last_output=np.argmax(output[0,i])
	return sent
	




# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

model_path='rnn_model.h5'
rnn_model=tf.keras.models.load_model(model_path)



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName('MainWindow')
        MainWindow.resize(1370, 780)
        self.lastEditPage=''
        [self.dict_id,self.dict_data]=load_data()
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName('centralwidget')
        self.imgFrame = QtWidgets.QFrame(self.centralwidget)
        self.imgFrame.setGeometry(QtCore.QRect(680, 60, 640, 640))
        self.imgFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.imgFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.imgFrame.setObjectName('imgFrame')
        self.dataChoice = QtWidgets.QComboBox(self.centralwidget)
        self.dataChoice.setGeometry(QtCore.QRect(120, 70, 121, 22))
        self.dataChoice.setObjectName('dataChoice')
        self.dataChoice.addItem('')
        self.dataChoice.addItem('')
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 70, 121, 20))
        self.label.setObjectName('label')
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 10, 331, 50))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.label_2.setFont(font)
        self.label_2.setObjectName('label_2')
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(80, 100, 41, 21))
        self.label_3.setObjectName('label_3')
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 130, 131, 20))
        self.label_4.setObjectName('label_4')
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(120, 100, 121, 20))
        self.lineEdit.setObjectName('lineEdit')
        self.lineEdit.setText('0')
        self.datasetCaption = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.datasetCaption.setEnabled(True)
        self.datasetCaption.setReadOnly(True)
        self.datasetCaption.setGeometry(QtCore.QRect(10, 150, 341, 121))
        self.datasetCaption.setObjectName('datasetCaption')
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(680, 20, 131, 31))
        self.label_5.setObjectName('label_5')
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(20, 360, 111, 21))
        self.label_6.setObjectName('label_6')
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(40, 290, 291, 50))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.label_7.setFont(font)
        self.label_7.setObjectName('label_7')
        self.captionImage = QtWidgets.QPushButton(self.centralwidget)
        self.captionImage.setGeometry(QtCore.QRect(400, 390, 161, 23))
        self.captionImage.setObjectName('captionImage')
        self.imagePath = QtWidgets.QLineEdit(self.centralwidget)
        self.imagePath.setReadOnly(True)
        self.imagePath.setEnabled(True)
        self.imagePath.setGeometry(QtCore.QRect(140, 360, 321, 20))
        self.imagePath.setObjectName('imagePath')
        self.generatedCaption = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.generatedCaption.setEnabled(True)
        self.generatedCaption.setReadOnly(True)
        self.generatedCaption.setGeometry(QtCore.QRect(20, 420, 541, 71))
        self.generatedCaption.setObjectName('generatedCaption')
        self.openImage = QtWidgets.QPushButton(self.centralwidget)
        self.openImage.setGeometry(QtCore.QRect(480, 360, 81, 23))
        self.openImage.setObjectName('openImage')
        self.listWidgetPages  = QtWidgets.QListWidget(self.centralwidget)
        self.listWidgetPages.setGeometry(QtCore.QRect(380, 100, 171, 171))
        self.listWidgetPages.setObjectName('listWidgetPages ')
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 169, 169))
        self.scrollAreaWidgetContents.setObjectName('listWidgetPagesContents')
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName('statusbar')
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName('toolBar')
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        ########################  events  ######################## 
        self.captionImage.clicked.connect(self.captionImg_fct)
        self.openImage.clicked.connect(self.openImg_fct)
        self.dataChoice.currentIndexChanged.connect(self.setSelectedType_fct)
        self.lineEdit.textChanged.connect(self.choisirPage_fct)
        self.listWidgetPages.currentRowChanged.connect(self.setSelectedItem)
        self.load_fct()
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate('MainWindow', 'Description d\'image'))
        self.dataChoice.setItemText(0, _translate('MainWindow', 'Entrainement'))
        self.dataChoice.setItemText(1, _translate('MainWindow', 'Validation'))
        self.label.setText(_translate('MainWindow', 'Type de données :'))
        self.label_2.setText(_translate('MainWindow', 'Exploitation du dataset'))
        self.label_3.setText(_translate('MainWindow', 'Page :'))
        self.label_4.setText(_translate('MainWindow', 'Les  légendes :'))
        self.label_5.setText(_translate('MainWindow', 'L\'image :'))
        self.label_6.setText(_translate('MainWindow', 'Le chemin de l\'image :'))
        self.label_7.setText(_translate('MainWindow', 'Description d\'image'))
        self.captionImage.setText(_translate('MainWindow', 'Description d\'image'))
        self.openImage.setText(_translate('MainWindow', 'Ouvrir'))
        self.toolBar.setWindowTitle(_translate('MainWindow', 'toolBar'))
        
      ########################  events functions ######################## 
    def load_fct(self):
        print('load_fct')
        data_type=getSelectedType(self.dataChoice.currentIndex())
        num_page=int(self.lineEdit.text())
        page_data=get_page(data_type,num_page,self.dict_id)
        self.listWidgetPages.clear()
        for data in page_data:
            item=QtWidgets.QListWidgetItem()
            item.setText(data)                        
            self.listWidgetPages.addItem(item)
        
        
    def captionImg_fct(self):
        print('captionImg_fct')
        image_path=self.imagePath.text()
        if(image_path==''):
            self.generatedCaption.clear() 
            self.generatedCaption.moveCursor(QtGui.QTextCursor.End,QtGui.QTextCursor.MoveAnchor)
            self.generatedCaption.insertPlainText('Le chemin de l\'image est vide')

        (image_edited,features)=get_features(detection_model, image_path)
        path='temp/temp.jpg'
        #rescaled = (255.0 / image_edited.max() * (image_edited- image_edited.min())).astype(np.uint8)
        im = Image.fromarray(image_edited)
        im.save(path)
        self.imgFrame.setStyleSheet('background-image: url(\"'+path+'")')
        caption=predict_sent( rnn_model, features)
        self.generatedCaption.clear() 
        self.generatedCaption.moveCursor(QtGui.QTextCursor.End,QtGui.QTextCursor.MoveAnchor)
        self.generatedCaption.insertPlainText(caption)
        print('-',caption,'-')
        
    def openImg_fct(self):
        print('openImg_fct')
        filename = QtWidgets.QFileDialog.getOpenFileName()
        path = filename[0]
        print(path)
        if(path.endswith('jpg')):
            image_np = np.array(Image.open(path))
            img_height,img_width,_=image_np.shape
            self.imgFrame.setGeometry(QtCore.QRect(680, 60,img_width,img_height))
            self.imgFrame.setStyleSheet('background-image: url(\"'+path+'")')
            self.imagePath.setText(path)
        else:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setIcon(QtWidgets.QMessageBox.Critical)
            msgBox.setText('Veillez choisir une image')
            msgBox.setWindowTitle('Erreur de type du fichier')
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msgBox.exec()
        
    def setSelectedType_fct(self):
        print('set selected type')
        self.lineEdit.setText('0')
        self.load_fct()
        
    def choisirPage_fct(self):
        print('set page edit') 
        data_type=getSelectedType(self.dataChoice.currentIndex())
        page=isValidPage(data_type,self.dict_id,self.lineEdit.text())
        if(page==0):
            self.lineEdit.setText('0')
        if(page==-1):
            self.lineEdit.setText(self.lastEditPage)
        else:
            self.lineEdit.setText(str(page))
        self.lastEditPage=self.lineEdit.text()
        self.load_fct()
        
    def setSelectedItem(self,index):
        if index==-1 :
            return 
        print('selectItem_fct')
        num_page=int(self.lineEdit.text())
        img_index=num_page*10+index
        img_type=getSelectedType(self.dataChoice.currentIndex())
        img_id=self.dict_id[img_type][img_index]
        [img_width,img_height,img_url,img_captions,_]=self.dict_data[img_type][img_id].values()
        text='L\'indexe '+str(img_index)
        for i in range(0,len(img_captions)):
            text+='\n'+str(i+1)+':'+img_captions[i]
        
        if not os.path.exists('temp'):
            os.makedirs('temp')
        temp_path='./temp/'+str(img_id)+'.jpg'
        self.imagePath.setText(temp_path)
        if not path.exists(temp_path):
            urllib.request.urlretrieve(img_url,temp_path)
        self.datasetCaption.clear() 
        self.datasetCaption.moveCursor(QtGui.QTextCursor.End,QtGui.QTextCursor.MoveAnchor)
        self.datasetCaption.insertPlainText(text)
        self.imgFrame.setGeometry(QtCore.QRect(680, 60,img_width,img_height))
        self.imgFrame.setStyleSheet('background-image: url(\"'+temp_path+'")')
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


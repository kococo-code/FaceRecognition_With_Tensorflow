import glob 
import os 
import tensorflow as tf 
import cv2 
import numpy as np 
def MakeDataset():
    ImgFormat = ['jpg','png','jpeg']
    TargetDir = 'media/face'
    DatasetDir = 'media/lfw/lfw'
    #Anchor , Positive, Negative
    dataset = []
    label = []
    negDataset = glob.glob(DatasetDir+'/*/*.jpg')
    for i in os.listdir(TargetDir):
        
        targetDataset = glob.glob(TargetDir+'/'+i+'/*.jpg')
        anchor = targetDataset[0]
        for j in range(1,len(targetDataset)):
            for negImg in negDataset:
                dataset.append([anchor,targetDataset[j],negImg])
                label.append(i)
    return dataset ,label

def ExportTFRecord(dataset,label):
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def image_example(anchor,positive,negative, label):
        Anchor = tf.image.decode_jpeg(anchor).shape
        Postive = tf.image.decode_jpeg(positive).shape
        Negative = tf.image.decode_jpeg(negative).shape
        feature = {
            'label': _int64_feature(label),
            'Anchor': _bytes_feature(Anchor),
            'Positive' : _bytes_feature(Postive),
            'Negative': _bytes_feature(Negative)
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))
    export_fileName = 'images.tfrecords'
    dataset = []
    with tf.io.TFRecordWriter(export_fileName) as writer:
        for i in range(len(dataset)):
            anchor = open(dataset[i][0], 'rb').read()
            postive = open(dataset[i][1], 'rb').read()
            negative = open(dataset[i][2], 'rb').read()

            tf_example = image_example(anchor,positive,negative, label)
            writer.write(tf_example.SerializeToString())



dataset, label = MakeDataset()
ExportTFRecord(dataset,label)
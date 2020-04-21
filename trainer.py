import tensorflow as tf 
from tensorflow.keras.layers import Dense, Input ,Conv2D, MaxPool2D,BatchNormalization,Flatten,Dropout
from tensorflow.keras import Model 
#Anchor Positive Negative
class DeepFace():
    def __init__(self,num_classes):
        self.numclasses = num_classes
        self.input_shape= (250,250,3)
        self.dropoutRate = 0.3
        self.batch_size = 32
        self.shuffle_buffer = 1000
    def Img2Vec(self,input_tensor,scope):
        # InputTensor inputShape
        model =tf.keras.applications.InceptionResNetV2(input_tensor = input_tensor, input_shape=self.input_shape,name=scope)
        return model
    def VGG19(self,inputTensor,nameScope):
        def bn_Conv2D(filters=1,kernel_size=1,strides=1,padding="SAME",activation='relu',nameSubScope=None,input_tensor=None):
            ConvLayer = Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                activation=activation,
                name=nameScope+'_'+nameSubScope)(input_tensor)
            BnNorm = BatchNormalization()(ConvLayer)
            return BnNorm
        conv1_1 =  bn_Conv2D(64,3,nameSubScope="Conv1_1",input_tensor=inputTensor)
        conv1_2 = bn_Conv2D(64,3,nameSubScope="Conv1_2",input_tensor=conv1_1)
        pool1 = MaxPool2D(pool_size=2,strides=2,name=nameScope+'_pool_1')(conv1_2)
        conv2_1 = bn_Conv2D(128,3,nameSubScope="Conv2_1",input_tensor=pool1)
        conv2_2 = bn_Conv2D(128,3,nameSubScope="Conv2_2",input_tensor=conv2_1)
        pool2 = MaxPool2D(pool_size=2,strides=2,name=nameScope+'_pool_2')(conv2_2)
        conv3_1 = bn_Conv2D(256,3,nameSubScope="Conv3_1",input_tensor=pool2)
        conv3_2 = bn_Conv2D(256,3,nameSubScope="Conv3_2",input_tensor=conv3_1)
        conv3_3 = bn_Conv2D(256,3,nameSubScope="Conv3_3",input_tensor=conv3_2)
        conv3_4 = bn_Conv2D(256,3,nameSubScope="Conv3_4",input_tensor=conv3_3)
        pool3 = MaxPool2D(pool_size=2,strides=2,name=nameScope+'_pool_3')(conv3_4)
        conv4_1 = bn_Conv2D(512,3,nameSubScope="Conv4_1",input_tensor=pool3)
        conv4_2 = bn_Conv2D(512,3,nameSubScope="Conv4_2",input_tensor=conv4_1)
        conv4_3 = bn_Conv2D(512,3,nameSubScope="Conv4_3",input_tensor=conv4_2)
        conv4_4 = bn_Conv2D(512,3,nameSubScope="Conv4_4",input_tensor=conv4_3)
        pool4 = MaxPool2D(pool_size=2,strides=2,name=nameScope+'_pool_4')(conv4_4)
        conv5_1 = bn_Conv2D(512,3,nameSubScope="Conv5_1",input_tensor=pool4)
        conv5_2 = bn_Conv2D(512,3,nameSubScope="Conv5_2",input_tensor=conv5_1)
        conv5_3 = bn_Conv2D(512,3,nameSubScope="Conv5_3",input_tensor=conv5_2)
        conv5_4 = bn_Conv2D(512,3,nameSubScope="Conv5_4",input_tensor=conv5_3)
        pool5 = MaxPool2D(pool_size=2,strides=2,name=nameScope+'_pool_5')(conv5_4)
        flatLayer = Flatten()(pool5)

        fc6 = Dense(4096,activation='relu')(flatLayer)
        dropout1 = Dropout(rate=self.dropoutRate)(fc6)
        fc7 = Dense(4096,activation='relu')(dropout1)
        dropout2 = Dropout(rate=self.dropoutRate)(fc7)
        fc8 = Dense(128,activation='relu')(dropout2)
        return fc8
    def model(self):
        # Make Embedding 
        Anchor = Input(self.input_shape)
        Positive = Input(self.input_shape)
        Negative = Input(self.input_shape)
        Anchor_Embedding= self.VGG19(Anchor,'Anchor')
        Positive_Embedding= self.VGG19(Positive,'Positive')
        Negative_Embedding= self.VGG19(Negative,'Negative')
        embedding = Model(inputs=[Anchor,Positive,Negative],outputs=[Anchor_Embedding,Positive_Embedding,Negative_Embedding])

        return embedding

    def createDataset(self,filepath):
        def _parse_function(proto):
            # define your tfrecord again. Remember that you saved your image as a string.
            keys_to_features = {'Anchor': tf.io.FixedLenFeature([], tf.string),
                                'Positive': tf.io.FixedLenFeature([], tf.string),
                                'Negative': tf.io.FixedLenFeature([], tf.string),
                                "label": tf.io.FixedLenFeature([], tf.int64)}
            
            # Load one example
            parsed_features = tf.io.parse_single_example(proto, keys_to_features)
            
            # Turn your saved image string into an array
            parsed_features['Anchor'] = tf.io.decode_raw(parsed_features['Anchor'], tf.uint8)
            parsed_features['Positive'] = tf.io.decode_raw(parsed_features['Positive'], tf.uint8)
            parsed_features['Negative'] = tf.io.decode_raw(parsed_features['Negative'], tf.uint8)

            return parsed_features['Anchor'],parsed_features['Positive'],parsed_features['Negative'], parsed_features["label"]
        dataset = tf.data.TFRecordDataset(filepath)
        
        dataset = dataset.map(_parse_function, num_parallel_calls=8)
        
        dataset = dataset.repeat()
        
        dataset = dataset.shuffle(self.shuffle_buffer)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        image, label = iterator.get_next()

        # Bring your picture back in shape
        #image = tf.reshape(image, [-1, 256, 256, 1])
        
        # Create a one hot array for your labels
        label = tf.one_hot(label, self.num_classes)
        
        return image, label
    def trainer(self):
        load_tfrecord = self.createDataset('./images.tfrecords')
        model = self.model()
        model.compile(optimizer=self.optimzier,loss=tfa.losses.triplet_semihard_loss)
        model.fit(load_tfrecord,epoches=1)        

DeepFace(3).trainer()
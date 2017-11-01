from keras.layers import *
from keras.models import *
from keras.optimizers import *
from tqdm import tqdm
import h5py

class TextCNN():
    def __init__(self, title_len=25,content_len=1500,seq_len=100):
        self.title_len=title_len
        self.content_len=content_len
        self.seq_len=seq_len
        self.model=self.CNN_model()
        self.optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='binary_crossentropy',optimizer=self.optimizer,metrics=['accuracy'])

    def CNN_model(self):
        #title
        input_title = Input(shape=(self.title_len, self.seq_len))

        x_title1 = Conv1D(64, 4, strides=1, padding='valid',activation='relu',input_shape=(self.title_len, self.seq_len))(input_title)
        # x_title1 = BatchNormalization()(x_title1)
        x_title1 = GlobalMaxPooling1D()(x_title1)

        x_title2 = Conv1D(64, 3, strides=1, padding='valid',activation='relu',input_shape=(self.title_len, self.seq_len))(input_title)
        # x_title2 = BatchNormalization()(x_title2)
        x_title2 = GlobalMaxPooling1D()(x_title2)

        x_title3 = Conv1D(64, 2, strides=1, padding='valid',activation='relu',input_shape=(self.title_len, self.seq_len))(input_title)
        # x_title3 = BatchNormalization()(x_title3)
        x_title3 = GlobalMaxPooling1D()(x_title3)

        #content
        input_content = Input(shape=(self.content_len, self.seq_len))

        x_content1 = Conv1D(128, 5, strides=1, padding='valid',activation='relu',input_shape=(self.content_len, self.seq_len))(input_content)
        # x_content1 = BatchNormalization()(x_content1)
        x_content1 = GlobalMaxPooling1D()(x_content1)

        x_content2 = Conv1D(128, 4, strides=1, padding='valid',activation='relu',input_shape=(self.content_len, self.seq_len))(input_content)
        # x_content2 = BatchNormalization()(x_content2)
        x_content2 = GlobalMaxPooling1D()(x_content2)

        x_content3 = Conv1D(128, 3, strides=1, padding='valid',activation='relu',input_shape=(self.content_len, self.seq_len))(input_content)
        # x_content3 = BatchNormalization()(x_content3)
        x_content3 = GlobalMaxPooling1D()(x_content3)

        x_content4 = Conv1D(128, 2, strides=1, padding='valid', activation='relu',input_shape=(self.content_len, self.seq_len))(input_content)
        # x_content4 = BatchNormalization()(x_content4)
        x_content4 = GlobalMaxPooling1D()(x_content4)

        x = Concatenate()([x_title1,x_title2,x_title3,x_content1,x_content2,x_content3,x_content4])
        # x = BatchNormalization()(x)
        # x = Dropout(0.5)(x)
        # x = Dense(2048,activation='relu')(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='sigmoid')(x)
        return Model(inputs=[input_title,input_content], outputs=output, name='Discriminator')

    def fit(self,path,batch_size=64,epochs=5,eval=0,random_seed=2017,shuffle=True,patience=1):
        np.random.seed(random_seed)
        train_list=np.array(os.listdir(path))
        if shuffle:
            np.random.shuffle(train_list)
        if eval!=0:
            eval_list=train_list[0:int(len(train_list)*eval)]
            train_list=train_list[int(len(train_list)*eval):]
        self.path=path
        self.epochs=epochs
        self.lower_best = 0
        self.batch_index=None
        self.best_eval_acc=0
        for i in range(epochs):
            j=0
            one_epoch=False
            while not one_epoch:
                print('epoch '+str(i+1)+': '+str((j + 1) * batch_size)+'/'+str(len(train_list)))
                if (j + 1) * batch_size <= len(train_list):
                    titles, contents, labels = self.train_generator(train_list[j * batch_size:(j + 1) * batch_size])
                else:
                    titles, contents, labels = self.train_generator(train_list[j * batch_size:])
                    one_epoch=True
                    if shuffle:
                        np.random.shuffle(train_list)
                loss_acc=self.model.train_on_batch([titles,contents], labels)
                print('training loss/acc='+str(loss_acc))
                j+=1
            if eval!=0:
                eval_acc=self.evaluation(eval_list,eval_batch=100)
                print('eval acc=' + str(eval_acc))
                if eval_acc>self.best_eval_acc:
                    self.best_eval_acc=eval_acc
                    self.lower_best=0
                else:
                    self.lower_best+=1
                    if self.lower_best>patience:
                        print('stop!')
                        break
        return self

    def train_generator(self,train_list):
        titles=[]
        contents=[]
        labels=[]
        for s in train_list:
            file=  h5py.File(self.path + "/" + s, 'r')
            titles.append(self.seq_padding(file['title'],self.title_len))
            contents.append(self.seq_padding(file['content'],self.content_len))
            labels.append(self.label_process(file['label'].value))
        titles=np.reshape(titles, (len(train_list), self.title_len, self.seq_len))
        contents=np.reshape(contents,(len(train_list), self.content_len, self.seq_len))
        labels=np.array(labels)
        return titles,contents,labels

    def label_process(self,label):
        if label=='POSITIVE':
            return 1
        else:
            return 0

    def seq_padding(self,data,maxlen):
        if data.shape[0]>maxlen:
            return data[0:maxlen,:]
        elif data.shape[0]==0:
            return np.zeros((maxlen,self.seq_len))
        elif data.shape[0]<maxlen:
            ex_data=np.zeros((maxlen-data.shape[0],self.seq_len))
            data=np.concatenate((data,ex_data),axis=0)
        return data

    def predict(self,path,predict_batch=1000):
        test_list = np.array(os.listdir(path))
        self.test_path=path
        ids=[]
        preds=[]
        for i in range(int(len(test_list)/predict_batch)):
            print(str((i + 1)*predict_batch) + '/' + str(len(test_list)))
            if (i + 1) * predict_batch <= len(test_list):
                id, titles, contents = self.test_generator(test_list[i * predict_batch:(i + 1) * predict_batch])
            else:
                id, titles, contents = self.test_generator(test_list[i * predict_batch:])
            ids.extend(list(id))
            pred = self.model.predict_on_batch([titles, contents])
            preds.append(pred)
        return np.reshape(np.array(ids),(-1,1)),np.concatenate(preds,axis=0)

    def test_generator(self,test_list):
        id = []
        titles = []
        contents=[]
        for s in test_list:
            file = h5py.File(self.test_path + "/" + s, 'r')
            id.append(s.strip('.h5'))
            titles.append(self.seq_padding(file['title'], self.title_len))
            contents.append(self.seq_padding(file['content'], self.content_len))
        titles = np.reshape(titles, (len(test_list), self.title_len, self.seq_len))
        contents = np.reshape(contents, (len(test_list), self.content_len, self.seq_len))
        id = np.array(id)
        return id, titles, contents

    def evaluation(self,eval_list,eval_batch):
        true_label=[]
        preds=[]
        for i in range(int(len(eval_list) / eval_batch)):
            print('eval:'+str((i + 1) * eval_batch) + '/' + str(len(eval_list)))
            if (i + 1) * eval_batch <= len(eval_list):
                titles, contents, labels = self.train_generator(eval_list[i * eval_batch:(i + 1) * eval_batch])
            else:
                titles, contents, labels = self.train_generator(eval_list[i * eval_batch:])
            true_label.extend(list(labels))
            pred = self.model.predict_on_batch([titles, contents])
            preds.append(pred)
        preds=np.concatenate(preds,axis=0)
        preds[preds>0.5]=1
        preds[preds<=0.5]=0
        preds=np.reshape(preds,(1,-1))[0]
        true_label=np.array(true_label)
        true_label=preds-true_label
        return len(np.nonzero(true_label==0)[0])/len(true_label)
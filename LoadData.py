import numpy as np
import struct
import matplotlib.pyplot as plt
import os

class DataUtils( object ):
    """MNIST数据集加载
        输出格式为：numpy.array()

        使用方法如下
        from data_util import DataUtils
        def main():
            trainfile_X = '../dataset/MNIST/train-images.idx3-ubyte'
            trainfile_y = '../dataset/MNIST/train-labels.idx1-ubyte'
            testfile_X = '../dataset/MNIST/t10k-images.idx3-ubyte'
            testfile_y = '../dataset/MNIST/t10k-labels.idx1-ubyte'

            train_X = DataUtils(filename=trainfile_X).getImage()
            train_y = DataUtils(filename=trainfile_y).getLabel()
            test_X = DataUtils(testfile_X).getImage()
            test_y = DataUtils(testfile_y).getLabel()

            #以下内容是将图像保存到本地文件中
            #path_trainset = "../dataset/MNIST/imgs_train"
            #path_testset = "../dataset/MNIST/imgs_test"
            #if not os.path.exists(path_trainset):
            #    os.mkdir(path_trainset)
            #if not os.path.exists(path_testset):
            #    os.mkdir(path_testset)
            #DataUtils(outpath=path_trainset).outImg(train_X, train_y)
            #DataUtils(outpath=path_testset).outImg(test_X, test_y)

            return train_X, train_y, test_X, test_y
    """


    def __init__( self, fileName = None, outPath = None ):
        self._fileName = fileName
        self._outPath = outPath

        self._tag = '>'
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'
        self._pictureBytes = '784B'
        self._labelBytes = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes    #>IIII’指的是使用大端法读取4个unsinged int 32 bit integer
        self._pictureBytes2 = self._tag + self._pictureBytes    #>784B’指的是使用大端法读取784个unsigned byte
        self._labelBytes2 = self._tag + self._labelBytes


    def getImage( self ):
        """
        将MNIST的二进制文件转换成新书特征数据
        """
        binFile = open( self._fileName, 'rb' )    #以二进制的方式打开文件
        buffer = binFile.read()
        binFile.close()
        index = 0
        numMagic, numImages, numRows, numColumns = struct.unpack_from( self._fourBytes2, buffer, index )
        index += struct.calcsize( self._fourBytes )
        images = []
        for i in range( numImages ):
            imageValue = struct.unpack_from( self._pictureBytes2, buffer, index )     #struct.unpack_from( fmt, buffer, offser )从offset位置开始解包buffer，按照fmt格式输出
            index += struct.calcsize( self._pictureBytes2 )
            imageValue = list( imageValue )
            for j in range( len( imageValue ) ):
                if imageValue[j] > 1:
                    imageValue[j] = 1
            images.append( imageValue )
        return np.array( images )


    def getLabel( self ):
        """
        将MNIST中label二进制文件转化成对应的label数字特征
        """
        binFile = open( self._fileName, 'rb' )
        buffer = binFile.read()
        binFile.close()
        index = 0
        magic, numItems = struct.unpack_from( self._twoBytes2, buffer, index )
        index += struct.calcsize( self._twoBytes2 )
        labels = [];
        for x in range( numItems ):
            im = struct.unpack_from( self._labelBytes2, buffer, index )
            index += struct.calcsize( self._labelBytes2 )
            labels.append( im[0] )
        return np.array( labels )



    def outImage( self, arrX, arrY ):
        """
        根据生成的特征和数字标号，输出png图像
        """
        m, n = np.shape( arrX )
        # 每张图是28 * 28 = 784byte
        for i in range( 1 ):
            image = np.array( arrX[i] )
            image = image.reshape( 28, 28 )
            outFile = str( i ) + "_" + str( arrY[ i ] ) + '.png'
            plt.figure()
            plt.imshow( image, cmap = 'binary' ) #将图片黑白显示
            plt.savefig( self._outPath + '/' + outFile )
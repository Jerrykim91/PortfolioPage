# https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/mnist/beginners/

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')
mnist = input_data.read_data_sets('./data/mnist/', one_hot=True)

pixels = mnist.train.images.shape[1]  # 이미지 한개당 특징(feature)의 크기 
nums = mnist.train.labels.shape[1]    # 레이블 특징(feature)의 크기
pixel_wh = int( np.sqrt( pixels ) )   # 이미지 1개당 가로 혹은 세로 크기


x = tf.placeholder(tf.float32 , shape =(None, pixels), name = 'x' ) # 플레이스 홀더

# 가중치 필터 W를 만드는 함수 
def makeWeightVariable(shape, name):
    init_d = tf.truncated_normal(shape, stddev= 0.1) # 초기값
    W = tf.Variable(init_d, name = 'W_'+name) # W를 생성
    return W   # 가중치 필터(커널) W를 리턴

# 편향을 만드는 함수 
def makeBiasVariable(shape, name):

    init_b = tf.constant( 0.1, shape=[shape] ) 
    b      = tf.Variable( init_b, name='b_' + name )
    return b   # 편향 값을 리턴 받는다.

# 합성곱층을 만드는 함수 
def makeConv2d(x, W, name):
    conv2d = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding= "SAME", name = 'conv_'+name )
    return conv2d

# 풀링함수 -> 최대풀링 
def makeMaxPooling( x ):
    return tf.nn.max_pool( x, ksize = [1, 2, 2, 1], strides = [1 ,2, 2, 1], padding = 'SAME'  )

# 합성곱층 1 생성, 입력대비 출력까지의 모든 관계(그래프)를 표현 
with tf.name_scope('conv1') as scope:
    W = makeWeightVariable( [5, 5, 1, 32 ] , 'conv1' )
    b = makeBiasVariable(32 , 'conv1')  # b
    # x : 입력층 (None, 784) => (batch(배치), h(세로), w(가로), channel(채널))
    x_imgs  = tf.reshape(x, (-1, pixel_wh, pixel_wh, 1))
    h_conv1 = tf.nn.relu( makeConv2d( x_imgs, W,'conv1') + b )  # conv1

# 풀링층 1 생성 
with tf.name_scope('pool1') as scope:
    h_pool1 = makeMaxPooling( h_conv1 )

# 합성곱층 2  생성 
name_conv2 = 'conv2'
with tf.name_scope(name_conv2) as scope:
    # [5(h), 5(w), 32(입력채널수=이전단계의출력채널수), 64(최종 출력채널수)]
    W_conv2 = makeWeightVariable( [5, 5, 32, 64 ], name_conv2 )
    b_conv2 = makeBiasVariable( 64, name_conv2 )
    h_conv2 = tf.nn.relu( makeConv2d( h_pool1, W_conv2, name_conv2 ) + b_conv2 )

# 풀링층 2
with tf.name_scope('pool2') as scope:
    h_pool2 = makeMaxPooling( h_conv2 )

# 전 결합층 
with tf.name_scope('fully_conected') as scope:
    num = 7 * 7 * 64 
    W_flat = makeWeightVariable([num, 1024],'fully_conected')
    h_pool2_flat = tf.reshape(h_pool2,[-1, num])
    h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_flat ))

# 드롭아웃층 
with tf.name_scope('dropout') as scope:
    keep_prob = tf.placeholder(tf.float32)
    h_fc_drop = tf.nn.dropout(h_fc, rate=1-keep_prob)

# 출력 층 
with tf.name_scope('output') as scope:
    # nums = 10
    W_output = makeWeightVariable( [1024,nums], 'output' )   # W
    b_output = makeBiasVariable( nums, 'output' )         # b

    # y_conv => tf.nn.softmax( tf.matmul(x,w)+b)
    # x => h_fc_drop
    y_conv = tf.nn.softmax( tf.matmul(h_fc_drop,W_output)+b_output)


y_ = tf.placeholder(tf.float32, shape=(None, nums), name = 'y_') # 정답


# 크로스 엔트로피 
with tf.name_scope('loss') as scope:
    cross_entropy = - tf.reduce_sum(y_ * tf.log(y_conv))

# 경사하강법 
with tf.name_scope('agd') as scope:
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize( cross_entropy )

# 예측, 평가 관련 플로우작성 
with tf.name_scope('predict') as scope:
    predict = tf.equal(tf.arg_max(y_conv, 1 ),tf.arg_max(y_, 1 ))  # 예측
    accuracy = tf.reduce_mean(tf.cast(predict, tf.float32)) # 정확도 

# 주입할 데이터의 모양을 세팅해주는 함수를 구성 
def makeFeedDictParam(imgDatas, labels, prob):
    return { x: imgDatas , y_: labels , keep_prob:prob } 

# 실전
TRAIN_COUNTS = 3000      # 설정값
ONE_TRAIN_COUNTS = 50    # 한번 훈련시 사용하는 데이터의 양 
VERBOSE_TERM = 100       # 100번째 훈련이 되면 로그를 출력 


with tf.Session() as sess: 
    init = tf.global_variables_initializer()
    sess.run(init)  # 초기화 
    test_img = mnist.test.images
    test_lab = mnist.test.labels
    keep_prob_size  = 1
    test_feedDict = makeFeedDictParam( test_img, test_lab, keep_prob_size)

    for step in range(TRAIN_COUNTS):   # 0~2999 : 3천번 수행 
        batch = mnist.train.next_batch(ONE_TRAIN_COUNTS)  #  0:이미지 데이터, 1:레이블정답 데이터
        train_fdp = makeFeedDictParam( batch[0],  batch[1], 0.5)
        _, loss =sess.run([train, cross_entropy],feed_dict= train_fdp )

        if step % VERBOSE_TERM == 0 :
            acc = sess.run( accuracy, feed_dict = test_feedDict )
            print('s=%4s, a=%20s, l=%20s' % (step, acc, loss) ) 

    acc = sess.run( accuracy, feed_dict = test_feedDict )
    print('-'*50)
    print('최종 결과')
    print( 's=%4s, a=%20s, l=%20s' % (step, acc, loss) )
    print('-'*50)
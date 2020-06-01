import tensorflow as tf
import numpy as np

def deeplab_v3(X,filter_cnt=32,OS=8,training=False,class_num=1,minmax='minmax',train_cls=False):
    he_init = tf.variance_scaling_initializer(scale=2.0)
    l2_reg = tf.contrib.layers.l2_regularizer(0.001)
    def conv2d(x,filter,dilate_rate,training,name,kernel=3,stride=1):
       x = tf.layers.conv2d(
           inputs=x,
           filters=filter,
           kernel_size=(kernel, kernel),
           strides=(stride,stride),
           activation=None,
           padding='same',
           kernel_initializer=he_init,
           kernel_regularizer=l2_reg,
           name="conv_{}".format(name + 1),
           dilation_rate=(dilate_rate,dilate_rate))
       x = tf.layers.batch_normalization(
           x, training=training, name="bn_{}".format(name))
       x = tf.nn.relu(x, name="relu_{}".format(name))

       return x

    def seperable_conv2d(inputs,filters,dilate_size,training,name,kernel_size=3,stride=1):
       x = tf.layers.separable_conv2d(inputs=inputs,filters=filters,
                                  kernel_size=(kernel_size,kernel_size),
                                  strides=(stride,stride),
                                  padding='same',
                                  dilation_rate=(dilate_size,dilate_size),
                                  depthwise_initializer=he_init,
                                  pointwise_initializer=he_init,
                                  depthwise_regularizer=l2_reg,
                                  pointwise_regularizer=l2_reg,
                                  # trainable=training,
                                  name="sep_{}".format(name)
                                  )
       x = tf.layers.batch_normalization(
           x, training=training, name="bn_{}".format(name))
       x = tf.nn.relu(x, name="relu_{}".format(name))
       return x

    # def xception_block(inputs,):
    if OS == 8:
       entry_block3_stride = 1
       middle_block_rate = 2  # ! Not mentioned in paper, but required
       exit_block_rates = (2, 4)
       atrous_rates = (12, 24, 36)

    else:
       entry_block3_stride = 2
       middle_block_rate = 1
       exit_block_rates = (1, 2)
       atrous_rates = (6, 12, 18)

    if minmax == 'minmax':
       net = X / 255.0
    elif minmax == 'std':
       net = X / 127.5 - 1
    elif minmax == 'no':
       net = X
    print(minmax)

    net = conv2d(net,filter=filter_cnt,dilate_rate=1,stride=2,training=training,name=1)
    print(net.shape, filter_cnt)
    net = conv2d(net,filter=filter_cnt*2,dilate_rate=1,stride=1,training=training,name=2)
    print(net.shape)

    # 1 xception block

    residual = conv2d(x=net,filter=filter_cnt*4,dilate_rate=1,stride=2,training=training,name=3)
    for i in range(3):
       sep_stride = 2
       net = seperable_conv2d(inputs=net,filters=filter_cnt*4,
                        stride=sep_stride if i == 2 else 1,kernel_size=3,dilate_size=1,
                        training=training,name=40+i)
       print(net.shape)

    net = tf.add(net,residual)
    print(net.shape)

    # 2 xception block
    residual = conv2d(x=net, filter=filter_cnt * 8, dilate_rate=1, stride=2, training=training, name=5)
    for i in range(3):
       sep_stride = 2
       net = seperable_conv2d(inputs=net, filters=filter_cnt * 8,
                              stride=sep_stride if i == 2 else 1, kernel_size=3, dilate_size=1,
                              training=training, name=50+i)
       print(net.shape)

       if i == 1:
           skip_connetion = net

    net = tf.add(net, residual)
    print(net.shape)

    # 3 xception block
    print('3 xception')
    residual = conv2d(x=net, filter=728, dilate_rate=1, stride=entry_block3_stride, training=training, name=7)
    for i in range(3):
       sep_stride = entry_block3_stride
       print(sep_stride if i == 2 else 1)
       net = seperable_conv2d(inputs=net, filters=728,
                              stride=sep_stride if i == 2 else 1, kernel_size=3, dilate_size=1,
                              training=training, name=60+i)
       print(net.shape)

    net = tf.add(net, residual)
    print(net.shape)

    # 4 xception block
    print('4 xception')
    for i in range(16):
       residual = net
       net = seperable_conv2d(inputs=net, filters=728,
                              stride=1, kernel_size=3, dilate_size=middle_block_rate,
                              training=training, name=70+i)
       net = tf.add(net, residual)
    print(net.shape)

    # 5 xception block
    print('5 xception')
    residual = conv2d(x=net, filter=1024, dilate_rate=1, stride=1, training=training, name=10)
    for i in range(3):
       sep_stride = 1
       sep_filters = 728
       net = seperable_conv2d(inputs=net, filters=sep_filters if i == 0 else 1024,
                              stride=sep_stride if i == 2 else 1, kernel_size=3, dilate_size=exit_block_rates[0],
                              training=training, name=90+i)
       print(net.shape)

    net = tf.add(net, residual)
    print(net.shape)

    # 6 xception block
    print('6 xception')
    for i in range(3):
       sep_filters = 2048
       net = seperable_conv2d(inputs=net, filters=sep_filters if i == 2 else 1536,
                              stride=1, kernel_size=3, dilate_size=exit_block_rates[1],
                              training=training, name=100+i)
       print(net.shape)
    print(net.shape)

    if train_cls:
       cls1 = tf.keras.layers.GlobalAveragePooling2D()(net)
       cls2 = tf.keras.layers.Dropout(0.5)(cls1)
       cls3 = tf.keras.layers.Dense(2048, activation='relu')(cls2)
       cls4 = tf.keras.layers.Dense(2, activation='softmax')(cls3)

    # ASPP block

    # image pooling
    b4 = tf.keras.layers.AveragePooling2D(pool_size=(int(np.ceil(int(X.shape[1]) / OS)), int(np.ceil(int(X.shape[1]) / OS))))(net)
    b4 = conv2d(x=b4,filter=256,dilate_rate=1,stride=1,kernel=1,training=training,name=14)
    b4 = tf.keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x, size=(
    int(np.ceil(int(X.shape[1]) / OS)), int(np.ceil(int(X.shape[1]) / OS)))))(b4)

    # b4 = tf.keras.layers.AveragePooling2D(pool_size=(1,1))(net)
    # b4 = conv2d(x=b4,filter=256,dilate_rate=1,stride=1,training=trainable,name=14,kernel=1)

    # 1*1 conv
    b0 = conv2d(x=net,filter=256,dilate_rate=1,stride=1,kernel=1,training=training,name=15)

    b1 = seperable_conv2d(inputs=net,filters=256,stride=1,kernel_size=3,dilate_size=atrous_rates[0],training=training,name=16)
    b2 = seperable_conv2d(inputs=net, filters=256, stride=1, kernel_size=3, dilate_size=atrous_rates[1],
                         training=training, name=17)
    b3 = seperable_conv2d(inputs=net, filters=256, stride=1, kernel_size=3, dilate_size=atrous_rates[2],
                         training=training, name=18)

    print(b0.shape)
    print(b1.shape)
    print(b2.shape)
    print(b3.shape)

    net = tf.concat([b4,b0,b1,b2,b3],axis=3)
    print(net.shape)
    net = conv2d(x=net,filter=256,dilate_rate=1,stride=1,kernel=1,training=training,name=19)

    net = tf.keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x, size=(
    int(np.ceil(int(X.shape[1]) / 4)), int(np.ceil(int(X.shape[1]) / 4)))))(net)
    print(net.shape)

    skip_connetion = conv2d(x=skip_connetion,filter=48,dilate_rate=1,kernel=1,
                           stride=1,training=training,name=20)

    net = tf.concat([net,skip_connetion],axis=3)

    net = seperable_conv2d(inputs=net,filters=256,stride=1,kernel_size=3,dilate_size=1,training=training,name=21)
    net = seperable_conv2d(inputs=net, filters=256, stride=1, kernel_size=3, dilate_size=1, training=training, name=22)

    print(net.shape)

    net = conv2d(x=net,filter=class_num,dilate_rate=1,stride=1,kernel=1,training=training,name=23)

    net = tf.keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x, size=(int(X.shape[1]), int(X.shape[1]))))(net)
    # net = tf.nn.softmax(net)
    # net = tf.nn.sigmoid(net)

    print(net.shape)

    if train_cls:
        return net, cls4
    else:
        return net

def ce_net_conv(X, training, filter_cnt=64,flags=None,minmax='minmax'):
   kernel_init = tf.variance_scaling_initializer(scale=2.0)
   seg_num_class = 1

   def conv_conv_pool(input_,
                      n_filters,
                      training,
                      flags,
                      name,
                      pool=True,
                      activation=tf.nn.relu,
                      sep_bool=True,
                      stride=1,
                      kernel_size=3,
                      dilate=1):
       net = input_
       sep = sep_bool
       if sep:
           with tf.variable_scope("layer{}".format(name)):
               for i, F in enumerate(n_filters):
                   net = tf.layers.separable_conv2d(
                       net,
                       F, (kernel_size, kernel_size),
                       strides=stride,
                       activation=None,
                       padding='same',
                       depthwise_initializer=kernel_init,
                       pointwise_initializer=kernel_init,
                       depthwise_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                       pointwise_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                       name="sep_conv_{}".format(i))
                   net = tf.layers.batch_normalization(
                       net, training=training, name="bn_{}".format(i + 1))
                   net = activation(net, name="relu{}_{}".format(name, i + 1))

               if pool is False:
                   return net

               pool = tf.layers.max_pooling2d(
                   net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

           return net, pool
       else:
           with tf.variable_scope("layer{}".format(name)):
               for i, F in enumerate(n_filters):
                   net = tf.layers.conv2d(
                       net,
                       F, (kernel_size, kernel_size),
                       activation=None,
                       strides=stride,
                       padding='same',
                       dilation_rate=dilate,
                       kernel_initializer=kernel_init,
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                       name="conv_{}".format(i + 1))
                   net = tf.layers.batch_normalization(
                       net, training=training, name="bn_{}".format(i + 1))
                   net = activation(net, name="relu{}_{}".format(name, i + 1))

               if pool is False:
                   return net

               pool = tf.layers.max_pooling2d(
                   net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

           return net, pool

   def upconv_2D(tensor, n_filter, flags, name):

       return tf.layers.conv2d_transpose(
           tensor,
           filters=n_filter,
           kernel_size=2,
           strides=2,
           kernel_initializer=kernel_init,
           kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
           name="upsample_{}".format(name))

   def decoder_block(d_net,filter_cnt,training,flags,name):
       d_net = conv_conv_pool(d_net,[filter_cnt],training,flags,name='d_block' + name,pool=False,sep_bool=False,kernel_size=1)
       d_net = upconv_2D(d_net,filter_cnt,flags,'d_block' + name)
       d_net = tf.layers.batch_normalization(
           d_net, training=training, name="up_bn_{}".format(name))
       d_net = tf.nn.relu(d_net, name="up_relu{}".format(name))
       d_net = conv_conv_pool(d_net, [filter_cnt], training, flags, name='d_block_1' + name, pool=False, sep_bool=False,
                              kernel_size=1)
       return d_net

   if minmax == 'minmax':
       net = X / 255.0
   elif minmax == 'std':
       net = X / 127.5 - 1
   elif minmax == 'no':
       net = X
   print(minmax)

   print(net.shape)
   conv0 = conv_conv_pool(net, [filter_cnt], training, flags, pool=False, name='1',sep_bool=False,stride=2,kernel_size=7)
   # conv0 = tf.layers.conv2d(net, filter_cnt, 7, 2, 'same', kernel_initializer=kernel_init)
   print(conv0.shape)
   pool1 = tf.layers.max_pooling2d(conv0, (2, 2), strides=(2, 2), name="pool_{}".format('1'))
   print(pool1.shape)
   pool1 = conv_conv_pool(pool1, [filter_cnt], training, flags, pool=False,
                            name='1_r', sep_bool=False, kernel_size=1)

   conv1_1 = conv_conv_pool(pool1,[filter_cnt],training,flags,pool=False,name='1_1',sep_bool=False)
   conv1_2 = conv_conv_pool(conv1_1, [filter_cnt], training, flags, pool=False, name='1_2', sep_bool=False)

   conv2 = tf.add(pool1,conv1_2)
   print(conv2.shape)

   pool2 = tf.layers.max_pooling2d(conv2, (2, 2), strides=(2, 2), name="pool_{}".format('2'))
   conv2_r = conv_conv_pool(pool2, [filter_cnt * (2**1)], training, flags, pool=False, name='2_r', sep_bool=False,kernel_size=1)

   conv2_1 = conv_conv_pool(conv2_r, [filter_cnt * (2**1)], training, flags, pool=False, name='2_1', sep_bool=False)
   conv2_2 = conv_conv_pool(conv2_1, [filter_cnt * (2**1)], training, flags, pool=False, name='2_2',
                            sep_bool=False)

   conv3 = tf.add(conv2_r, conv2_2)
   print(conv3.shape)

   pool3 = tf.layers.max_pooling2d(conv3, (2, 2), strides=(2, 2), name="pool_{}".format('3'))
   conv3_r = conv_conv_pool(pool3, [filter_cnt * (2 ** 2)], training, flags, pool=False,
                            name='3_r', sep_bool=False, kernel_size=1)

   conv3_1 = conv_conv_pool(conv3_r, [filter_cnt * (2 ** 2)], training, flags, pool=False,
                            name='3_1', sep_bool=False)
   conv3_2 = conv_conv_pool(conv3_1, [filter_cnt * (2 ** 2)], training, flags, pool=False, name='3_2',
                            sep_bool=False)

   conv4 = tf.add(conv3_2, conv3_r)
   print(conv4.shape)

   pool4 = tf.layers.max_pooling2d(conv4, (2, 2), strides=(2, 2), name="pool_{}".format('4'))
   conv4_r = conv_conv_pool(pool4, [filter_cnt * (2 ** 3)], training, flags, pool=False,
                            name='4_r', sep_bool=False, kernel_size=1)

   conv4_1 = conv_conv_pool(conv4_r, [filter_cnt * (2 ** 3)], training, flags, pool=False,
                            name='4_1', sep_bool=False)
   conv4_2 = conv_conv_pool(conv4_1, [filter_cnt * (2 ** 3)], training, flags, pool=False,
                            name='4_2', sep_bool=False)

   conv5 = tf.add(conv4_2, conv4_r)
   print(conv5.shape)

   dac1 = conv_conv_pool(conv5, [filter_cnt * (2 ** 3)], training, flags, pool=False,
                            name='dac_1', sep_bool=False,dilate=1)

   dac2 = conv_conv_pool(conv5, [filter_cnt * (2 ** 3)], training, flags, pool=False,
                            name='dac_2', sep_bool=False,dilate=3)
   dac2_1 = conv_conv_pool(dac2, [filter_cnt * (2 ** 3)], training, flags, pool=False,
                         name='dac_2_1', sep_bool=False, dilate=1,kernel_size=1)

   dac3 = conv_conv_pool(conv5, [filter_cnt * (2 ** 3)], training, flags, pool=False,
                         name='dac_3', sep_bool=False, dilate=1)
   dac3_1 = conv_conv_pool(dac3, [filter_cnt * (2 ** 3)], training, flags, pool=False,
                         name='dac_3_1', sep_bool=False, dilate=3)
   dac3_2 = conv_conv_pool(dac3_1, [filter_cnt * (2 ** 3)], training, flags, pool=False,
                           name='dac_3_2', sep_bool=False, dilate=1,kernel_size=1)

   dac4 = conv_conv_pool(conv5, [filter_cnt * (2 ** 3)], training, flags, pool=False,
                         name='dac_4', sep_bool=False, dilate=1)
   dac4_1 = conv_conv_pool(dac4, [filter_cnt * (2 ** 3)], training, flags, pool=False,
                         name='dac_4_1', sep_bool=False, dilate=3)
   dac4_2 = conv_conv_pool(dac4_1, [filter_cnt * (2 ** 3)], training, flags, pool=False,
                           name='dac_4_2', sep_bool=False, dilate=5)
   dac4_3 = conv_conv_pool(dac4_2, [filter_cnt * (2 ** 3)], training, flags, pool=False,
                           name='dac_4_3', sep_bool=False, dilate=1,kernel_size=1)

   dac_add = dac1 + dac2_1 + dac3_2 + dac4_3

   cls1 = tf.keras.layers.GlobalAveragePooling2D()(dac_add)
   cls2 = tf.keras.layers.Dropout(0.5)(cls1)
   cls3 = tf.keras.layers.Dense(filter_cnt * (2 ** 4), activation='relu')(cls2)
   cls4 = tf.keras.layers.Dense(2, activation='softmax')(cls3)

   spp1 = tf.layers.max_pooling2d(dac_add, (2, 2), strides=(2, 2), name="pool_{}".format('spp_1'))
   spp2 = tf.layers.max_pooling2d(dac_add, (3, 3), strides=(3, 3), name="pool_{}".format('spp_2'))
   spp3 = tf.layers.max_pooling2d(dac_add, (5, 5), strides=(5, 5), name="pool_{}".format('spp_3'))
   spp4 = tf.layers.max_pooling2d(dac_add, (6, 6), strides=(6, 6), name="pool_{}".format('spp_4'))

   spp1 = conv_conv_pool(spp1, [1], training, flags, pool=False, name='spp1_1', sep_bool=False, kernel_size=1)
   spp2 = conv_conv_pool(spp2, [1], training, flags, pool=False, name='spp2_1', sep_bool=False, kernel_size=1)
   spp3 = conv_conv_pool(spp3, [1], training, flags, pool=False, name='spp3_1', sep_bool=False, kernel_size=1)
   spp4 = conv_conv_pool(spp4, [1], training, flags, pool=False, name='spp4_1', sep_bool=False, kernel_size=1)

   print(conv5.shape[1])
   spp1 = tf.image.resize_bilinear(spp1,size=(conv5.shape[1],conv5.shape[1]))
   spp2 = tf.image.resize_bilinear(spp2,size=(conv5.shape[1],conv5.shape[1]))
   spp3 = tf.image.resize_bilinear(spp3,size=(conv5.shape[1],conv5.shape[1]))
   spp4 = tf.image.resize_bilinear(spp4,size=(conv5.shape[1],conv5.shape[1]))

   spp_concat = tf.concat([dac_add,spp1,spp2,spp3,spp4],axis=3)
   print(spp_concat.shape)

   up1 = decoder_block(spp_concat,256,training,flags,'up_1')
   print(up1.shape)
   print(conv3_r.shape)

   up1 = tf.concat([up1,conv3_r],axis=3)
   print(up1.shape)

   up2 = decoder_block(up1,128,training,flags,'up_2')
   up2 = tf.concat([up2,conv2_r],axis=3)

   up3 = decoder_block(up2, 64, training, flags, 'up_3')
   up3 = tf.concat([up3, conv1_2], axis=3)

   up4 = decoder_block(up3, 64, training, flags, 'up_4')
   up4 = tf.concat([up4, conv0], axis=3)

   heatmap_output = decoder_block(up4, seg_num_class, training, flags, 'up_5')
   heatmap_output = tf.nn.sigmoid(heatmap_output)
   print(heatmap_output.shape)

   return heatmap_output, cls4


if __name__ == '__main__':
    X = tf.placeholder(shape=(None, 300, 300, 3), dtype=tf.float32)
    logits = deeplab_v3_with_cls(X,filter_cnt=32,training=False)

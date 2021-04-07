import tensorflow as tf

def tversky_focal_loss(y_true, y_pred,alpha=0.7,gamma=0.75, focal=True):
   epsilon = 1e-5
   smooth = 1
   def tversky(y_true, y_pred):
       y_true_pos = tf.keras.layers.Flatten()(y_true)
       y_pred_pos = tf.keras.layers.Flatten()(y_pred)
       true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
       false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
       false_pos = tf.reduce_sum((1-y_true_pos)*y_pred_pos)
       # alpha = 0.4 # 0.7
       return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

   # def tversky_loss(y_true, y_pred):
   #     return 1 - tversky(y_true,y_pred)
   if focal:
       pt_1 = tversky(y_true, y_pred)
       # gamma = 0.9
       return K.pow((1-pt_1), gamma)
   else:
       return 1 - tversky(y_true, y_pred)

def focal_loss_sigmoid(labels,logits,alpha=0.1,gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `labels`
    """
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
    L=-labels*(1-alpha)*((1-y_pred)*gamma)*tf.log(y_pred)-\
      (1-labels)*alpha*(y_pred**gamma)*tf.log(1-y_pred)
    L= tf.reduce_mean(L)
    return L

def focal_loss_softmax(labels,logits,gamma=2,alpha=0.25):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `labels`
    """
    y_pred=tf.nn.softmax(logits,dim=-1) # [batch_size,num_classes]
    labels=tf.one_hot(labels,depth=y_pred.shape[1])
    L=-labels*(1-alpha)*((1-logits)**gamma)*tf.log(logits)
    print(L.shape)
    L= tf.reduce_mean(tf.reduce_sum(L))
    return L

def focal_loss(y_true,y_pred,alpha=0.13, gamma=2):
   def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
       weight_a = alpha * (1 - y_pred) ** gamma * targets
       weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

       return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

   y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
   logits = tf.log(y_pred / (1 - y_pred))

   loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

   return tf.reduce_mean(loss)


def cls_loss_function(y_true,y_pred):
    # cost = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true,y_pred))
    cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=y_pred))
    return cost

def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize

def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")

def loss_function(y_true,y_pred,weights=1.0):
    # cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred))
    # cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(y_pred,y_true))

    # y_pred = tl.act.pixel_wise_softmax(y_pred)
    # y_true = tl.act.pixel_wise_softmax(y_true)
    # cost = 1 - tl.cost.dice_coe(y_pred, y_true)
    # cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

    # cost = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true,y_pred))

    def pixel_wise_softmax(output_map):
        with tf.name_scope("pixel_wise_softmax"):
            max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
            exponential_map = tf.exp(output_map - max_axis)
            normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
            return exponential_map / normalize

    def cross_entropy(y_, output_map):
        return -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name="cross_entropy")

    # y_pred = tf.reshape(y_pred, [-1, 2])
    # y_true = tf.reshape(y_true, [-1, 2])
    # cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_true,y_pred,pos_weight=weights))
    cost = tf.keras.losses.binary_crossentropy(y_true,y_pred)
    # cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(y_true,y_pred))
    return cost

def balanced_cross_entropy(y_pred,y_true,beta):
  def convert_to_logits(y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

      return tf.log(y_pred / (1 - y_pred))

  # def loss(y_true, y_pred):
  y_pred = convert_to_logits(y_pred)
  pos_weight = beta / (1 - beta)
  loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # or reduce_sum and/or axis=-1
  loss = tf.reduce_mean(loss * (1 - beta))

  return loss

def weighted_loss(y_true,y_pred,weights=1.0):
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_true,y_pred,pos_weight=weights))
    return cost

def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.
    #
    # Examples
    # ---------
    # >>> import tensorlayer as tl
    # >>> outputs = tl.act.pixel_wise_softmax(outputs)
    # >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)
    #
    # References
    # -----------
    # - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    # old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    # new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice

if __name__ == '__main__':
    logits=tf.random_uniform(shape=[5],minval=-1,maxval=1,dtype=tf.float32)
    labels=tf.Variable([0,1,0,0,1])
    loss1=focal_loss_sigmoid(labels=labels,logits=logits)

    logits2=tf.random_uniform(shape=[5,4],minval=-1,maxval=1,dtype=tf.float32)
    labels2=tf.Variable([1,0,2,3,1])
    loss2=focal_loss_softmax(labels==labels2,logits=logits2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(loss1))
        print(sess.run([loss2,logits2]))

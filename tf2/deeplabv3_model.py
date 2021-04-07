# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import random, imutils
import matplotlib.pyplot as plt
import os, math
import tensorflow as tf
import cv2, time
from only_model_advanced import *
from image_util import *
from loss_function import *

def run_model(gpu=1,
              train_path='',
              val_path='',
              img_size=320,
              epoch_cnt=100,
              batch_num=2,
              filter_num=32,
              save_dir='',
              select_model='',
              normalize_method='minmax',
              loss_multiply=1,
              cls_loss_method='focal',
              seg_loss_method='focal',
              alpha_value=0.2,
              gamma_value=0.9,
              class_weights=10.0,
              source_img='',
              ):

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    do_ne = False
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('gpus', gpus)
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    input_size = img_size
    seed_list = [random.randint(1,100) for x in range(10)]
    ckpt_path = '/media/cres04/DATA/fracture/segmentation_result/ckpt/proto_end/unet.ckpt'
    pretrain_use = False

    batch_size = batch_num
    dump = False
    seg_num_class = 1
    filter_cnt = filter_num
    save_path = save_dir + '/'
        # '/media/j/DATA/fracture/segmentation_frac/result/scaphoid_dense_sep_conv_minmax_focal_weight*3' + '/'
    # save_ckpt = save_path + 'unet.ckpt'
    for_save = True
    use_l2_loss = False
    X_start = normalize_method
    if 'deeplab_v3' == select_model:
        model_name = deeplab_v3
    elif 'deeplab_v3_with_cls' == select_model:
        model_name = deeplab_v3_with_cls

    def makedir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    makedir(save_path)
    f = open(save_path + 'parameter.txt', 'w')
    f.write('model_name : ' + str(select_model) + '\n')
    f.write('filter_cnt : ' + str(filter_cnt) + '\n')
    f.write('img_size : ' + str(seg_num_class) + '\n')
    f.write('batch_size : ' + str(batch_size) + '\n')
    f.write('input_size : ' + str(input_size) + '\n')
    f.write('alpha_value : ' + str(alpha_value) + '\n')
    f.write('gamma_value : ' + str(gamma_value) + '\n')
    # f.write('learning_rate : ' + str(learning_rate) + '\n')
    # f.write('batch_size : ' + str(batch_size) + '\n')
    # f.write('l2_rate : ' + str(l2_rate) + '\n')
    # f.write('drop_out : ' + str(drop_out) + '\n')
    # f.write('class_weight : ' + str(class_weight_bool) + '\n')
    # f.write('category : ' + str(category) + '\n')
    f.close()

    def set_category(path,category_list):
        train_img_path = path + '/'
        category = [train_img_path + x for x in category_list]
        mask_list = [x + '_mask' for x in category]

        return category, mask_list

    if do_ne == 'multi':
        train_img_path, train_mask_path = set_category(train_path, ['negative','positive'])
        val_img_path, val_mask_path = set_category(val_path, ['negative','positive'])
    else:
        train_img_path, train_mask_path = set_category(train_path, ['img'])
        val_img_path, val_mask_path = set_category(val_path, ['img'])


    # print(train_img_path)
    # print(val_img_path)

    def my_load(img_path,mask_path,aug):
        img_data = []
        mask_data = []

        for one_img, one_mask in zip(img_path,mask_path):
            img = cv2.imread(one_img)
            mask = cv2.imread(one_mask,0)

            if dump:
                dump_mask = (255-mask)
                dump_mask = np.expand_dims(dump_mask, axis=2)
                mask = np.expand_dims(mask,axis=2)
                mask = np.concatenate((mask,dump_mask),axis=2)
            else:
                mask = np.expand_dims(mask, axis=2)

            if aug:
                image_aug = random.randint(0, 4)
                if image_aug == 0:
                    pass
                elif image_aug == 1:
                    img = clahe(img)
                elif image_aug == 2:
                    img = normalize(img)
                elif image_aug == 3:
                    img = clahe(img)
                    img = sharpen(img)
                elif image_aug == 4:
                    img = gaussian_noise(img)
                elif image_aug == 5:
                    # radius '/media/j/DATA/fracture/detection_result/fix_professor_data/kah_crop/radius_SR/train/negative/00011497_F760000_6.png'
                    sour = source_img
                    img = histogram_match(img,sour)

                con_img = np.concatenate((img,mask),axis=2)

                flip = random.randint(0, 1)
                rotation = random.randint(0, 20)

                h_size = con_img.shape[0]
                w_size = con_img.shape[1]
                h_rsize = random.randint(np.floor(0.9 * h_size), h_size)
                w_rsize = random.randint(np.floor(0.9 * w_size), w_size)

                w_s = random.randint(0, w_size - w_rsize)
                h_s = random.randint(0, h_size - h_rsize)

                con_img = imutils.rotate(con_img,rotation)
                # #  Scale
                # con_img = transform.rescale(con_img, scale=np.random.uniform(0.8, 1.2),
                #                         mode='edge')

                # sh = random.random() / 2 - 0.25
                # rotate_angel = random.random() / 180 * np.pi * 20
                # Create Afine transform
                # afine_tf = transform.AffineTransform(shear=sh, rotation=rotate_angel)
                # Apply transform to image data
                # con_img = transform.warp(con_img, inverse_map=afine_tf, mode='edge')
                # label = transform.warp(label, inverse_map=afine_tf, mode='edge')
                # Randomly corpping image frame

                if flip:
                    con_img = con_img[:, ::-1, :]

                img = con_img[h_s:h_s + h_size, w_s:w_s + w_size, :3]
                mask = con_img[h_s:h_s + h_size, w_s:w_s + w_size, 3:]

                if normalize_method == 'minmax':
                    img = img / 255.0
                elif normalize_method == 'std':
                    img = img / 127.5 - 1
                elif normalize_method == 'no':
                    img = img
                # print(normalize_method)

            img = cv2.resize(img, (input_size, input_size))
            mask = cv2.resize(mask, (input_size, input_size))
            mask = np.expand_dims(mask, axis=2)

            img_data.append(np.array(img))
            mask_data.append(np.array(mask))

        return np.array(img_data), np.array(mask_data)

    def list_load(img_path,mask_path):
        img_list = sorted(os.listdir(img_path))
        mask_list = sorted(os.listdir(mask_path))

        img_list = [img_path + '/' + x for x in img_list]
        mask_list = [mask_path + '/' + x for x in mask_list]

        return img_list, mask_list

    if do_ne == 'multi':
        ne_X_train, ne_Y_train = list_load(train_img_path[0], train_mask_path[0])
        po_X_train, po_Y_train = list_load(train_img_path[1], train_mask_path[1])
        X_train = ne_X_train + po_X_train
        Y_train = ne_Y_train + po_Y_train

        ne_X_val, ne_Y_val = list_load(val_img_path[0], val_mask_path[0])
        po_X_val, po_Y_val = list_load(val_img_path[1], val_mask_path[1])

        X_val = ne_X_val + po_X_val
        Y_val = ne_Y_val + po_Y_val
    else:
        po_X_train, po_Y_train = list_load(train_img_path[0], train_mask_path[0])
        X_train = po_X_train
        Y_train = po_Y_train

        po_X_val, po_Y_val = list_load(val_img_path[0], val_mask_path[0])

        X_val = po_X_val
        Y_val = po_Y_val

        # print(X_train,X_val)

    # X_train, Y_train = my_load('/media/j/DATA/fracture/segmentation_frac/radius/train/frac',
    #                            '/media/j/DATA/fracture/segmentation_frac/radius/train/frac_mask')
    # n_X_train, n_Y_train = my_load('/media/j/DATA/fracture/segmentation_frac/radius/train/ne',
    #                            '/media/j/DATA/fracture/segmentation_frac/radius/train/ne_mask')

    val_total_cnt = len(X_val)
    train_total_cnt = len(X_train)

    def shuffle():
        seed_int = random.choice(seed_list)
        np.random.seed(seed_int)
        np.random.shuffle(X_train)
        np.random.seed(seed_int)
        np.random.shuffle(Y_train)

    def next_batch(data_list, mini_batch_size, next_cnt):
        cnt = mini_batch_size * next_cnt
        batch_list = data_list[cnt:cnt + mini_batch_size]
        return batch_list

    # logits = UNet(X_)
    # logits = Neural(X_ ,3,True,10).neural_net()
    # logits,logits2,logits3,logits4,logits5, cls_logits = sep_conv_make_unet(X_,trainable)
    # logits, cls_logits = sep_conv_make_unet(X_,trainable)
    input_layer = tf.keras.layers.Input(shape=(img_size,img_size, 3))
    output_layer = model_name(input_layer, minmax=X_start)
    output_layer.trainable = True
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

    if pretrain_use:
        model.load_weights(ckpt_path)

        print('load pre-trained_weight')

    def iou_coe(output, target, threshold=0.5, smooth=1e-5):
        axis = [1, 2, 3]
        pre = tf.cast(output > threshold, dtype=tf.float32)
        truth = tf.cast(target > threshold, dtype=tf.float32)
        inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
        union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
        batch_iou = (inse + smooth) / (union + smooth)
        iou = tf.reduce_mean(batch_iou)
        return iou

    # def mean_iou(y_pred,y_true):
    #     y_pred_ = tf.to_int64(y_pred > 0.5)
    #     y_true_ = tf.to_int64(y_true > 0.5)
    #     score, up_opt = tf.metrics.mean_iou(y_true_, y_pred_,1)
    #     with tf.control_dependencies([up_opt]):
    #         score = tf.identity(score)
    #     return score

    # custom loss
    # self.output = tl.act.pixel_wise_softmax(out_seg)

    # self.loss = 1 - tl.cost.dice_coe(self.output, self.Y)

    # self.loss = tl.cost.dice_hard_coe(self.output, self.Y)

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_seg, labels=self.Y))

    #self.l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    #self.loss = tf.add(self.loss,self.l2_loss)



    # cross_entropy = cross_entropy(tf.reshape(Y_, [-1, 1]),
    #                                        tf.reshape(pixel_wise_softmax(logits), [-1, 1]))
    #
    # predicter = pixel_wise_softmax(logits)

    # specify the weights for each sample in the batch (without having to compute the onehot label matrix)
    # weights = tf.gather(class_weights, Y_)

    # loss_heat_map = seg_loss_func(Y_, logits, class_weights) * loss_multiply
    # loss_heat_map2 = loss_function(Y_, logits2)
    # loss_heat_map3 = loss_function(Y_, logits3)
    # loss_heat_map4 = loss_function(Y_, logits4)
    # loss_heat_map5 = loss_function(Y_, logits5)
    # loss_cls = focal_loss_softmax(cY_, cls_logits)

    # loss_heat_map = tf.reduce_sum(loss_heat_map + loss_heat_map2 + loss_heat_map3 + loss_heat_map4
    #                                              + loss_heat_map5 + loss_cls)

    # if use_l2_loss:
    #     reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #     add_total_loss += tf.reduce_sum(reg)

    # out_logits = tf.nn.sigmoid(logits)

    # out_logits, _ = tf.split(out, [1, 1], 3)
    # print(out_logits.shape)
    # out_Y, _ = tf.split(Y_, [1, 1], 3)
    # print(Y_.shape)

    # predicter = pixel_wise_softmax(logits)
    # correct_pred = tf.equal(tf.argmax(predicter, 3), tf.argmax(Y_, 3))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    display_count = 1
    best_val_iou = 0
    total_train_loss = 0
    total_train_iou = 0
    val_iou_dict = {}

    total_train_heatmap_loss_list = []
    total_val_heatmap_loss_list = []
    total_train_loss_list = []

    train_batch_cnt = math.ceil(train_total_cnt // batch_size)
    val_batch_cnt = math.ceil(val_total_cnt // batch_size)

    for epoch in range(epoch_cnt):
        start = time.time()
        for i in range(train_batch_cnt):

            batch_X = next_batch(X_train, batch_size,i)
            batch_Y = next_batch(Y_train, batch_size,i)

            # print(batch_X)
            # print(batch_Y)
            # print(batch_cY)

            batch_X, batch_Y = my_load(batch_X,batch_Y, True)
            # print('batch_X.shape',batch_X.shape)
            # print('batch_Y.shape',batch_Y.shape)

            # plt.imshow(batch_X[0])
            # plt.show()
            with tf.GradientTape() as tape:
                Y_ = model(batch_X)
                if 'focal' == seg_loss_method:
                    train_loss = focal_loss(batch_Y, Y_, alpha=alpha_value, gamma=gamma_value) * loss_multiply
                elif 'softmax' == seg_loss_method:
                    train_loss = loss_function(batch_Y, Y_) * loss_multiply
                elif 'focal_tver' == seg_loss_method:
                    train_loss = tversky_focal_loss(batch_Y, Y_, alpha=alpha_value, gamma=gamma_value)
                elif 'weighted_loss' == seg_loss_method:
                    train_loss = weighted_loss(batch_Y, Y_, weights=class_weights) * loss_multiply

            grad = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

            train_iou_score = np.round(iou_coe(Y_, batch_Y),5)
            total_train_iou += train_iou_score
            train_loss = np.round(np.mean(train_loss),5)
            total_train_loss += train_loss

            # print('train_iou_score : ', round(train_iou_score,5), ' / train_loss : ', round(train_loss,5))
            print('train_iou_score : ', train_iou_score, ' / train_loss : ', train_loss, )

        tti = round(total_train_iou / train_batch_cnt,5)
        ttl = round(total_train_loss / train_batch_cnt,5)
        total_train_heatmap_loss_list.append(tti)
        total_train_loss_list.append(ttl)
        print('total train_iou_score : ', tti , ' / total train_loss : ', ttl)

        total_val_loss, total_val_iou = 0, 0

        for i in range(val_batch_cnt):
        # if (i % 10 == 0) and not i == 0:
            # show img
            # for q,c in zip(predict,y):
            #     # a = np.squeeze(i,axis=2)
            #     # a = np.cast[np.uint8](a)
            #
            #     print(q.shape)
            #     out = np.squeeze(q,axis=2)
            #     out1 = np.squeeze(c,axis=2)
            #     # _, out = cv2.threshold(out,0.5,255,cv2.THRESH_BINARY)
            #     # _, out1 = cv2.threshold(out1, 0.5, 255, cv2.THRESH_BINARY)
            #     # a = i * 255
            #     plt.imshow(out1)
            #     plt.show()
            #     plt.imshow(out)
            #     plt.show()

            # val_iter = (val_total_cnt // batch_size) + 1

            val_batch_X = next_batch(X_val, batch_size, i)
            val_batch_Y = next_batch(Y_val, batch_size, i)
            val_batch_X, val_batch_Y = my_load(val_batch_X, val_batch_Y, False)

            val_y_ = model(val_batch_X)
            val_loss = np.round(np.mean(loss_function(val_batch_Y, val_y_) * loss_multiply),5)
            total_val_loss += val_loss
            val_iou_score = np.round(iou_coe(val_y_, val_batch_Y),5)
            total_val_iou += val_iou_score

            # show img
            #     for q,c,z in zip(val_predict,val_y,val_batch_X):
                    # a = np.squeeze(i,axis=2)
                    # a = np.cast[np.uint8](a)

                    # print(q.shape)
                    # pred_y = np.squeeze(q,axis=2)
                    # true_y = np.squeeze(c,axis=2)
                    # z = z[:,:,0]
                    # print(z.shape)
                    # _, out = cv2.threshold(out,0.5,255,cv2.THRESH_BINARY)
                    # _, out1 = cv2.threshold(out1, 0.5, 255, cv2.THRESH_BINARY)
                    # a = i * 255
                    # plt.imshow(z)
                    # plt.show()
                    # plt.imshow(true_y)
                    # plt.show()
                    # plt.imshow(pred_y)
                    # plt.show()

                    # cv2.imwrite('/media/j/DATA/fracture/segmentation_frac/result/img/' + str(cnt) + '.png',z)
                    # cv2.imwrite('/media/j/DATA/fracture/segmentation_frac/result/y_true/' + str(cnt) + '.png', true_y * 255)
                    # cv2.imwrite('/media/j/DATA/fracture/segmentation_frac/result/y_pred/' + str(cnt) + '.png', pred_y * 255)
                    # cnt += 1

            print('val_iou_score : ', val_iou_score, ' / val_loss : ', val_loss)

        vti = round(total_val_iou / val_batch_cnt,5)
        vtl = round(total_val_loss / val_batch_cnt,5)

        total_val_heatmap_loss_list.append(vti)
        print('total val_iou_score : ', vti, ' / total val_loss : ', vtl)

        if for_save:
            if best_val_iou < vti:
                best_val_iou = vti
                makedir(save_path + 'iou')
                model.save_weights(save_path + 'iou' + '/' + str(display_count) + '_' + str(best_val_iou) + '_unet')
                print('save iou model')

        val_iou_dict[vti] = display_count
        print(val_iou_dict)

        # print('val classification loss : ', val_total_c_lo / ((val_total_cnt // batch_size) + 1))

        end = time.time()
        train_time = end - start

        # print(np.argmax(c_lo,axis=1))
        # print(np.argmax(cy,axis=1))

        # loss plot
        plt.plot(total_train_heatmap_loss_list, marker='', color='blue', label="train_iou" if epoch == 0 else "")
        plt.plot(total_val_heatmap_loss_list, marker='', color='red', label="val_iou" if epoch == 0 else "")
        plt.plot(total_train_loss_list, marker='', color='green', label="train_loss" if epoch == 0 else "")

        plt.legend()

        plt.savefig(save_path + 'plot.png')

    print("Done!")

if __name__ == '__main__':
        run_model(gpu=1,
                  train_path='/app2/train',
                  val_path='/app2/val',
                  img_size=320,
                  epoch_cnt=100,
                  batch_num=2,
                  filter_num=32,
                  save_dir='/app2/OA_seg',
                  select_model='deeplab_v3',
                  normalize_method='minmax',
                  loss_multiply=1,
                  cls_loss_method='softmax',
                  seg_loss_method='softmax',
                  alpha_value=0.2,
                  gamma_value=0.9,
                  class_weights=10.0,
                  source_img='',)

"""
DOCSTRING
"""
import numpy
import operator
import os
import random
import sys
import tensorflow
import time
import transform
import utils
import vgg 

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'

def _tensor_size(tensor):
    """
    DOCSTRING
    """
    return reduce(operator.mul, (d.value for d in tensor.get_shape()[1:]), 1)

def optimize(
    content_targets,
    style_target,
    content_weight,
    style_weight,
    tv_weight,
    vgg_path,
    epochs=2,
    print_iterations=1000,
    batch_size=4,
    save_path='saver/fns.ckpt',
    slow=False,
    learning_rate=1e-3,
    device='/cpu:0',
    debug=False,
    total_iterations=-1,
    base_model_path=None):
    """
    DOCSTRING
    """
    if slow:
        batch_size = 1
    mod = len(content_targets) % batch_size
    if mod > 0:
        print("Train set has been trimmed slightly..")
        content_targets = content_targets[:-mod] 
    style_features = {}
    batch_shape = (batch_size, 256, 256, 3)
    style_shape = (1,) + style_target.shape
    print(style_shape)
    # precompute style features
    print("Precomputing style features")
    sys.stdout.flush()
    with tensorflow.Graph().as_default(), tensorflow.device(device), tensorflow.Session(
        config=tensorflow.ConfigProto(allow_soft_placement=True)) as sess:
        style_image = tensorflow.placeholder(
            tensorflow.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        style_pre = numpy.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            features = numpy.reshape(features, (-1, features.shape[3]))
            gram = numpy.matmul(features.T, features) / features.size
            style_features[layer] = gram
    with tensorflow.Graph().as_default(), tensorflow.Session() as sess:
        X_content = tensorflow.placeholder(tensorflow.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)
        print("Precomputing content features")
        sys.stdout.flush()
        # precompute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]
        if slow:
            preds = tensorflow.Variable(tensorflow.random_normal(X_content.get_shape()) * 0.256)
            preds_pre = preds
        else:
            preds = transform.net(X_content/255.0)
            preds_pre = vgg.preprocess(preds)
        print("Building VGG net")
        sys.stdout.flush()
        net = vgg.net(vgg_path, preds_pre)
        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_loss = content_weight * (2 * tensorflow.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size)
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            size = height * width * filters
            feats = tensorflow.reshape(layer, (bs, height * width, filters))
            feats_T = tensorflow.transpose(feats, perm=[0,2,1])
            grams = tensorflow.batch_matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tensorflow.nn.l2_loss(grams - style_gram)/style_gram.size)
        style_loss = style_weight * reduce(tensorflow.add, style_losses) / batch_size
        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tensorflow.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tensorflow.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size
        loss = content_loss + style_loss + tv_loss
        # overall loss
        train_step = tensorflow.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tensorflow.initialize_all_variables())
        # if base model file is present, load that in to the session
        if base_model_path:
            saver = tensorflow.train.Saver()
            if os.path.isdir(base_model_path):
                ckpt = tensorflow.train.get_checkpoint_state(base_model_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    raise Exception("No checkpoint found...")
            else:
                saver.restore(sess, base_model_path)
        uid = random.randint(1, 100)
        print("UID: %s" % uid)
        sys.stdout.flush()
        for epoch in range(epochs):
            num_examples = len(content_targets)
            print("number of examples: %s" % num_examples)
            sys.stdout.flush()
            iterations = 0
            while iterations * batch_size < num_examples:
                print("Current iteration : %s" % iterations)
                sys.stdout.flush()
                start_time = time.time()
                curr = iterations * batch_size
                step = curr + batch_size
                X_batch = numpy.zeros(batch_shape, dtype=numpy.float32)
                for j, img_p in enumerate(content_targets[curr:step]):
                   X_batch[j] = utils.get_img(img_p, (256,256,3)).astype(numpy.float32)
                iterations += 1
                assert X_batch.shape[0] == batch_size
                feed_dict = {X_content: X_batch}
                train_step.run(feed_dict=feed_dict)
                end_time = time.time()
                delta_time = end_time - start_time
                if debug:
                    print("UID: %s, batch time: %s" % (uid, delta_time))
                is_print_iter = int(iterations) % print_iterations == 0
                if slow:
                    is_print_iter = epoch % print_iterations == 0
                is_last = False
                if epoch == epochs - 1 and iterations * batch_size >= num_examples:
                    is_last = True
                if total_iterations > 0 and iterations >= total_iterations:
                    is_last = True
                should_print = is_print_iter or is_last
                if should_print:
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]
                    test_feed_dict = {X_content: X_batch}
                    tup = sess.run(to_get, feed_dict = test_feed_dict)
                    _style_loss,_content_loss,_tv_loss,_loss,_preds = tup
                    losses = (_style_loss, _content_loss, _tv_loss, _loss)
                    if slow:
                        _preds = vgg.unprocess(_preds)
                    else:
                        saver = tensorflow.train.Saver()
                        res = saver.save(sess, save_path)
                    yield(_preds, losses, iterations, epoch)
                if is_last:
                    break

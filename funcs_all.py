# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:23:04 2021

@author: Ali
"""
#@tf.function
def remove_players(model, players):
    '''remove selected players in the Inception-v3 network.'''
    if isinstance(players, str):
        players = [players]
    for player in players:
        variables = layer_dic['_'.join(player.split('_')[:-1])]
        #var_vals = model.sess.run(variables)
        #var_vals = variables
        for var in variables:
            var_copy=var.numpy()
            if 'variance' in var.name:
                var_copy[..., int(player.split('_')[-1])] = 1.
            elif 'beta' in var.name:
                pass
            elif 'bias' in var.name:#ali modified. TODO: what to do with bias??
                pass
            else:
                var_copy[..., int(player.split('_')[-1])] = 0.
            #var.load(var, model.sess)
            var.assign(var_copy) #verified that it sets weights to zero in the model
        

def return_player_output(model, player):
    '''The output of a filter.'''
    layer = '_'.join(player.split('_')[:-1])
    layer = '/'.join(layer.split('/')[1:])
    unit = int(player.split('_')[-1])
    return model.ends[layer][..., unit]

#@tf.function
def one_iteration(
    model, 
    players,
    images, 
    labels, 
    chosen_players=None,
    c=None, 
    metric='accuracy',
    truncation=None
):
    '''One iteration of Neuron-Shapley algoirhtm.'''
    model.load_weights(filepath=CHECKPOINT)
    init_val = value(model, images, labels, metric)
    if c is None:
        c = {i: np.array([i]) for i in range(len(players))}
    elif not isinstance(c, dict):
        c = {i: np.where(c==i)[0] for i in set(c)}
    if truncation is None:
        truncation = len(c.keys())
    if chosen_players is None:
        chosen_players = np.arange(len(c.keys()))
    idxs = np.random.permutation(len(c.keys()))
    print(idxs)
    marginals = -np.ones(len(c.keys()))
    marginals[chosen_players] = 0.
    t = time.time()
    truncation_counter = 0
    old_val = init_val.copy()
    for n, idx in enumerate(idxs[::-1]):
        if idx in chosen_players:
            if old_val is None:
                old_val = value(model, images, labels, metric)
            remove_players(model, players[c[idx]])
            new_val = value(model, images, labels, metric)
            marginals[c[idx]] = (old_val - new_val) / len(c[idx])
            old_val = new_val
            if isinstance(truncation, int):
                if n >= truncation:
                    break
            else:
                if n%10 == 0:
                    print(n, time.time() - t, new_val)
                val_diff = new_val - base_value
                if metric == 'accuracy' and val_diff <= truncation:
                    truncation_counter += 1
                elif (metric == 'xe_loss' and
                      val_diff <= truncation * np.abs(base_value)):
                    truncation_counter += 1
                elif metric == 'binary' and val_diff <= truncation:
                    truncation_counter += 1
                elif metric == 'logit' and new_val <= truncation * np.abs(init_val):
                    truncation_counter += 1
                else:
                    truncation_counter = 0
                if truncation_counter > 5:
                    break
        else:
            old_val = None
            remove_players(model, players[c[idx]])        
    return idxs.reshape((1, -1)), marginals.reshape((1, -1))

#@tf.function
def sess_run(model, images, labels=None, batch_size=BATCH_SIZE):
    '''Divides inputs into smaller chunks and performs sess.run'''
    output = []    
    num_batches = int(np.ceil(len(images) / batch_size))
    #sess = tf.Session()
    
    for batch in range(num_batches):
        batch_idxs = np.arange(
            batch * batch_size, min((batch+1) * batch_size, len(images))
        )
        input_dic = images[batch_idxs]
        if labels is not None:
            if default:
                input_labels = labels[batch_idxs]-1
            else:
                input_labels = labels[batch_idxs]
        probs = model(input_dic)
        #probs = model.sess.run(model.output, input_dic)
        #probs = sess.run(model.output,input_dic)
        sample_accuracy = tf.cast(
            tf.math.equal(tf.math.argmax(probs, -1), 
                          input_labels), 
            tf.float32
        )
        accuracy = tf.reduce_mean(sample_accuracy)

        output.append(accuracy)
    try:
        return np.concatenate(output, 0)
    except:
        return np.array(output)
    
# def sess_run(model, images, labels=None, batch_size=BATCH_SIZE):
#     '''Divides inputs into smaller chunks and performs sess.run'''
#     output = []    
#     num_batches = int(np.ceil(len(images) / batch_size))
#     for batch in range(num_batches):
#         batch_idxs = np.arange(
#             batch * batch_size, min((batch+1) * batch_size, len(images))
#         )
#         input_dic = {model.input: images[batch_idxs]}
#         if labels is not None:
#             input_dic[model.y_input] = labels[batch_idxs]
#         output.append(model.sess.run(variable, input_dic))
#     try:
#         return np.concatenate(output, 0)
#     except:
#         return np.array(output)
    
def value(model, images, labels, metric='accuracy', batch_size=BATCH_SIZE):
    '''The performance of the model on given image-label pairs.'''
    if isinstance(labels, str):
        labels = np.array([model.label_to_id(labels)] * len(images))
    elif isinstance(labels, int):
        labels = np.array([labels] * len(images))
    num_batches = int(np.ceil(len(images) / batch_size))
    val = 0.
    if metric == 'accuracy':
        val = np.mean(sess_run(
            model,  images, labels, batch_size=batch_size))
    elif metric=='xe_loss':
        probs = sess_run(
            model, model.probs, images, labels, batch_size=batch_size)
        val = np.mean(np.log(probs[np.arange(len(probs)), labels]))
    elif metric=='binary':
        probs = sess_run(
            model, model.probs, images, labels, batch_size=batch_size)
        preds = np.argmax(probs, -1)
        key_labels = np.expand_dims(list(set(labels[labels != -1])), 0)
        corrects_1 = np.sum(labels[labels != -1] == preds[labels != -1])
        corrects_2 = np.sum(1 - np.equal(preds[labels == -1], key_labels))
        val = (corrects_1 + corrects_2) * 1. / len(preds)
    elif metric=='logit':
        logits = sess_run(
            model, model.ends['logits'], images, labels, batch_size=batch_size)
        class_logits = np.mean(logits[np.arange(len(logits)), labels])
        return class_logits
    else:
        raise ValueError('Invalid metric!')
    return val


def adversarial_attack(model, images, target_label, epsilon=16./255, iters=30, 
               norm='l_inf', perturb=False, delta=16./255/20,
               minval=0., maxval=1., batch_size=16):
    '''Creates iterative adversarial attacks with PGD.'''
    
    if isinstance(target_label, int):
        target_label = np.array([target_label] * len(images))
        
    def batch_attack(x, y, gradient):
        
        if not epsilon:
            return x
        x_hat = x.copy()
        if perturb:
            if norm == 'l_2':
                r = np.random.normal(size=x.shape)
                r_norm =  np.sqrt(np.sum(r ** 2, axis=tuple(np.arange(1, len(x.shape))), keepdims=True))
                x_hat += r / r_norm * delta
            elif norm == 'l_inf':
                x_hat += delta * np.sign(np.random.random(x.shape))
        for nu in range(iters):
            grd = model.sess.run(gradient, {
                model.input: x_hat, 
                model.y_input: y
            })
            if norm == 'l_2':
                grd_norm = np.sqrt(np.sum(grd ** 2, axis=tuple(np.arange(1, len(x.shape))), 
                                      keepdims=True))
                grd /= grd_norm + 1e-8
                prtrb = x_hat - grd * delta - x 
                prtrb_norm = np.sqrt(np.sum(prtrb ** 2, axis=tuple(np.arange(1, len(x.shape))),
                                            keepdims=True))
                prj_coef = (prtrb_norm <= epsilon) * 1. + (prtrb_norm > epsilon) * prtrb_norm / epsilon
                prtrb /= prj_coef
            elif norm == 'l_inf':
                #print(nu, np.max(np.abs(x - x_hat).reshape((-1, np.prod(x.shape[1:]))), -1),
                     #model.sess.run(model.loss, {model.input: x_hat, model.y_input: y}))
                grd = np.sign(grd)
                prtrb = x_hat - grd * delta - x 
                prtrb = np.clip(prtrb, -epsilon, epsilon)
            else:
                raise ValueError('Invalid Norm {}'.format(norm))
            x_hat = np.clip(x + prtrb, minval, maxval)
        return x_hat
    
    gradient = tf.gradients(model.loss, model.input)[0]
    x = []
    num_batches = int(np.ceil(len(images) / batch_size))
    for batch in range(num_batches):
        print(batch)
        batch_idxs = np.arange(
            batch * batch_size, min((batch+1) * batch_size, len(images))
        )
        x.append(batch_attack(images[batch_idxs], target_label[batch_idxs], gradient))
    return np.concatenate(x, 0)


def load_images(files_dir, filenames, num_workers=0):
    
    file_dirs = [os.path.join(files_dir, filename) for filename in filenames]
    if num_workers:
        pool = multiprocessing.Pool(num_workers)
        images = pool.map(lambda f: np.array(Image.open(f)), file_dirs)
        pool.close()
    else:
        images=[]
        for f in file_dirs:
            im = Image.open(f)
            if im.mode != 'RGB':
                im = im.convert('RGB')
            if not default: im=im.resize((224,224))            
            images.append(np.array(im))
        # images = [np.array(Image.open(f)) for f in file_dirs]
    if default:
        return (np.array(images)/255).astype('float32')
    else:
        images = preprocess_input(np.array(images))
        return images.astype('float32')



def return_target_images(image_list, key):
    
    all_classes = list(set([image.split('/')[-2] for image in image_list]))
    if key == 'all' or key == 'rnd' or key == '-all':
        target_classes = all_classes
    elif key[0] == '-':
        target_classes = list(set(all_classes) - set(key[1:].split('+')))
    else:
        target_classes = key.split('+')
    target_images = [filename for filename in image_list
                     if filename.split('/')[-2] in target_classes]
    return target_images


def make_adv_images(key, images, model):
    
    adv_images, adv_labels = [], []
    if key == '-all':
        adv_labels = np.random.choice(np.arange(1001), len(images))
        adv_images = adversarial_attack(model, images, adv_labels)
        return np.array(adv_images), np.array(adv_labels)
    keys = key[1:].split('+')
    num_label_imgs = len(images) // len(keys)
    for i, k in enumerate(keys):
        key_id = model.label_to_id(k)
        adv_labels.extend(key_id * np.ones(num_label_imgs).astype(int))
        label_imgs = images[i * num_label_imgs: (i+1) * num_label_imgs]
        adv_images.extend(adversarial_attack(model, label_imgs, key_id))
    return np.array(adv_images), np.array(adv_labels)


def load_images_labels(key, num_images, max_sample_size, model, max_size=25000):

    if default:
        image_list = open('val_images.txt').read().split('\n')[:max_size]
    else:
        image_list = open('val_images_CUB.txt').read().split('\n')[:max_size]        
    val_images = return_target_images(image_list, key)
    num_images = min(num_images, max_sample_size, len(val_images))
    filenames = np.random.choice(val_images, num_images, replace=False)
    images = load_images(
        os.path.join(DATA_DIR),
        filenames,
        0)
    if key[0] == '-':
        return make_adv_images(key, images, model)
    labels = np.array([val_labels.index(filename.split('/')[-2]) 
                       for filename in filenames])
    return images, labels

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:52:13 2018
gru for imputed data
@author: lyh
"""


from __future__ import print_function
import os
import numpy as np
from sklearn import metrics
import time
import mygru_cell
import tensorflow as tf
from tensorflow.python.ops import math_ops
tf.set_random_seed(1)   # set random seed
 
class grui(object):
    model_name = "GRU_I"
    def __init__(self, sess, args, dataset, test_set):
        self.lr = args.lr            
        self.sess=sess
        self.isbatch_normal=args.isBatch_normal
        self.isNormal=args.isNormal
        self.isSlicing=args.isSlicing
        self.dataset=dataset
        self.test_set = test_set
        self.epoch = args.epoch     
        self.batch_size = args.batch_size
        self.n_inputs = args.n_inputs                 # MNIST data input (img shape: 28*28)
        self.n_steps = dataset.maxLength                                # time steps
        self.n_hidden_units = args.n_hidden_units        # neurons in hidden layer
        self.n_classes = args.n_classes                # MNIST classes (0-9 digits)
        self.run_type=args.run_type
        self.result_path=args.result_path
        self.model_path=args.model_path
        self.log_dir=args.log_dir
        self.checkpoint_dir=args.checkpoint_dir
        self.num_batches = len(dataset.x) // self.batch_size
        # x y placeholder
        self.keep_prob = tf.placeholder(tf.float32) 
        self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.m = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.delta = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.mean = tf.placeholder(tf.float32, [self.n_inputs,])
        self.lastvalues = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.x_lengths = tf.placeholder(tf.int32,  shape=[self.batch_size,])
        # 对 weights biases 初始值的定义
       

    #concatenate x and m 
    #rth should be also concatenate after x, then decay the older state
    #rth's length is n_hidden_units



    def RNN(self,X, M, Delta,  Mean, Lastvalues, X_lengths,Keep_prob, reuse=False):
        #       2*3*2
        # X: batches * steps, n_inputs
        # m:batches * steps, n_inputs
        # delta:batches * steps, n_inputs
        # mean:n_inputs  mean of all observations, not contian the imputations
        # lastvalues: batches * steps, n_inputs  last obsevation value of x, if x is missing
        # if lastvalues is zero, take mean as it
        
         with tf.variable_scope("grui", reuse=reuse):
           
            # then wr_x should be transformed into a diag matrix:tf.matrix_diag(wr_x)
            wr_h=tf.get_variable('wr_h',shape=[self.n_inputs,self.n_hidden_units],initializer=tf.random_normal_initializer())
            w_out=tf.get_variable('w_out', shape=[self.n_hidden_units, self.n_classes],initializer=tf.random_normal_initializer())
        
            br_h=tf.get_variable('br_h', shape=[self.n_hidden_units, ],initializer=tf.constant_initializer(0.001))
            b_out=tf.get_variable('b_out', shape=[self.n_classes, ],initializer=tf.constant_initializer(0.001))
        
        
        
            Lastvalues=tf.reshape(Lastvalues,[-1,self.n_inputs])
            #M=tf.reshape(M,[-1,self.n_inputs])
            X = tf.reshape(X, [-1, self.n_inputs])
            Delta=tf.reshape(Delta,[-1,self.n_inputs])
            
            
            rth= tf.matmul( Delta, wr_h)+br_h
            rth=math_ops.exp(-tf.maximum(0.0,rth))
            
            #X = tf.reshape(X, [-1, n_inputs])
            #print(X.get_shape(),M.get_shape(),rth.get_shape())
            X=tf.concat([X,rth],1)
            
            X_in = tf.reshape(X, [-1, self.n_steps, self.n_inputs+self.n_hidden_units])
            
            #print(X_in.get_shape())
            # X_in = W*X + b
            #X_in = tf.matmul(X, weights['in']) + biases['in']
            # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
            #X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
         
            if "1.5" in tf.__version__ or "1.7" in tf.__version__ :   
                grud_cell = mygru_cell.MyGRUCell15(self.n_hidden_units)
            elif "1.4" in tf.__version__:
                grud_cell = mygru_cell.MyGRUCell4(self.n_hidden_units)
            elif "1.2" in tf.__version__:
                grud_cell = mygru_cell.MyGRUCell2(self.n_hidden_units)
            init_state = grud_cell.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
            outputs, final_state = tf.nn.dynamic_rnn(grud_cell, X_in, \
                                initial_state=init_state,\
                                sequence_length=X_lengths,
                                time_major=False)
         
            factor=tf.matrix_diag([1.0/9,1])
            tempout=tf.matmul(tf.nn.dropout(final_state,Keep_prob), w_out) + b_out
            results =tf.nn.softmax(tf.matmul(tempout,factor))    #选取最后一个 output
            #todo: dropout of 0.5 and batch normalization
            return results
    def build(self):
        
        self.pred = self.RNN(self.x, self.m, self.delta, self. mean, self.lastvalues, self.x_lengths, self.keep_prob)
        self.cross_entropy = -tf.reduce_sum(self.y*tf.log(self.pred))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)
         
        
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.saver = tf.train.Saver(max_to_keep=None)
        
        loss_sum = tf.summary.scalar("loss", self.cross_entropy)
        acc_sum = tf.summary.scalar("acc", self.accuracy)
        
        self.sum=tf.summary.merge([loss_sum, acc_sum])
        
        
    def model_dir(self,epoch):
        return "{}_{}_{}_{}_{}_{}/epoch{}".format(
            self.model_name, self.lr,
            self.batch_size, self.isNormal,
            self.isbatch_normal,self.isSlicing,
            epoch
            )
        
    def save(self, checkpoint_dir, step, epoch):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir(epoch), self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir, epoch):
        import re
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir(epoch), self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            #print(" [*] Failed to find a checkpoint")
            return False, 0
    
    def train(self):
        
        max_auc = 0.5
        model_dir2= "{}_{}_{}_{}_{}_{}".format(
            self.model_name, self.lr, 
            self.batch_size, self.isNormal,
            self.isbatch_normal,self.isSlicing
            )
        if not os.path.exists(os.path.join(self.checkpoint_dir, model_dir2)):
            os.makedirs(os.path.join(self.checkpoint_dir, model_dir2))
        result_file=open(os.path.join(self.checkpoint_dir, model_dir2, "result"),"a+")
        
        if os.path.exists(os.path.join(self.checkpoint_dir, self.model_dir(self.epoch), self.model_name)):
            for nowepoch in range(1,self.epoch+1):
                print(" [*] Load SUCCESS")
                print("epoch: "+str(nowepoch))
                self.load(self.checkpoint_dir,nowepoch)
                acc,auc,model_name=self.test(self.test_set,nowepoch)
                if auc > max_auc :
                    max_auc = auc 
                result_file.write("epoch: "+str(nowepoch)+","+str(acc)+","+str(auc)+"\r\n")
                print("")
            result_file.close()
            return max_auc 
        else:
            # initialize all variables
            tf.global_variables_initializer().run()
            counter = 1
            print(" [!] Load failed...")

        start_time=time.time()
        idx = 0
        epochcount=0
        dataset=self.dataset
        while epochcount<self.epoch:
            dataset.shuffle(self.batch_size,True)
            for data_x,data_y,data_mean,data_m,data_delta,data_x_lengths,data_lastvalues,_,_,_ in dataset.nextBatch():
                _,loss,summary_str,acc = self.sess.run([self.train_op,self.cross_entropy, self.sum, self.accuracy], feed_dict={\
                    self.x: data_x,\
                    self.y: data_y,\
                    self.m: data_m,\
                    self.delta: data_delta,\
                    self.mean: data_mean,\
                    self.x_lengths: data_x_lengths,\
                    self.lastvalues: data_lastvalues,\
                    self.keep_prob: 0.5})
        
                counter += 1
                idx+=1
            epochcount+=1
            idx=0
            #print("Optimization Finished! save model!")
            self.save(self.checkpoint_dir, counter, epochcount)

            acc,auc,model_name=self.test(self.test_set,epochcount)
            if auc > max_auc :
                max_auc = auc 
            result_file.write("epoch: "+str(epochcount)+","+str(acc)+","+str(auc)+"\r\n")
            print("")

        result_file.close()
        return max_auc 
        
    def test(self,dataset, epoch):
        start_time=time.time()
        counter=0
        dataset.shuffle(self.batch_size,False)
        totalacc=0.0
        totalauc=0.0
        auccounter=0
        for data_x,data_y,data_mean,data_m,data_delta,data_x_lengths,data_lastvalues,_,_,_ in dataset.nextBatch():
            summary_str,acc,pred = self.sess.run([self.sum, self.accuracy,self.pred], feed_dict={\
                self.x: data_x,\
                self.y: data_y,\
                self.m: data_m,\
                self.delta: data_delta,\
                self.mean: data_mean,\
                self.x_lengths: data_x_lengths,\
                self.lastvalues: data_lastvalues,\
                self.keep_prob: 1.0})
    
            try:
                auc = metrics.roc_auc_score(np.array(data_y),np.array(pred))
                totalauc+=auc
                auccounter+=1
                print("Batch: %4d time: %4.4f, acc: %.8f, auc: %.8f" \
                          % ( counter, time.time() - start_time, acc, auc))
            except ValueError:
                print("Batch: %4d time: %4.4f, acc: %.8f " \
                          % ( counter, time.time() - start_time, acc))
                pass
            totalacc+=acc
            counter += 1
        totalacc=totalacc/counter
        try:
            totalauc=totalauc/auccounter
        except:
            pass
        print("epoch is : %2.2f, Total acc: %.8f, Total auc: %.8f , counter is : %.2f , auccounter is %.2f" % (epoch, totalacc,totalauc,counter,auccounter))
        f=open(os.path.join(self.checkpoint_dir, self.model_dir(epoch), self.model_name,"final_acc_and_auc"),"w")
        f.write(str(totalacc)+","+str(totalauc))
        f.close()
        return totalacc,totalauc,self.model_name
  

# -*- coding: utf-8 -*-

import os
os.environ['CHAINER_TYPE_CHECK'] = '0'

import chainer
import util.general_tool as tool
from util.backend import Backend
from util.vocabulary import Vocabulary
from util.optimizer_manager import get_opt
from models.manager import get_model

#----------------------------------------------------------------------------
from collections import defaultdict

candidate_heads=defaultdict(set)
candidate_tails=defaultdict(set)
gold_heads = defaultdict(set)
gold_tails = defaultdict(set)
black_set = set()

tail_per_head=defaultdict(set)
head_per_tail=defaultdict(set)

train_data,dev_data,test_data=list(),list(),list()
trfreq = defaultdict(int)

glinks, grelations, gedges = 0,0,0

from more_itertools import chunked
import random

def rules_addition():
	tool.trace('load rules')
	global premise
	premise = defaultdict(set)
	for line in tool.read(args.rules_file):
		r, r_, flag = list(map(int,line.strip().split('\t')))
		premise[r].add((r_,flag))
	print(len(premise[0]))
	print(len(premise))

def initilize_dataset(): 
	global candidate_heads,gold_heads,candidate_tails,gold_tails
	global glinks, grelations, gedges
	global train_data,dev_data,test_data,trfreq

	# get properties of knowledge graph
	# 读取训练集
	tool.trace('load train')
	grelations = defaultdict(set)
	glinks = defaultdict(set)
	train_data = set()
	for line in tool.read(args.train_file):
		h,r,t = list(map(int,line.strip().split('\t')))
		# train_data包含了所有三元组
		train_data.add((h,r,t,))
		# trfreq键值为关系ID，值为有该关系的三元组数量
		trfreq[r]+=1
		# grelations存储了所有关系，键值为(头实体，尾实体)，值为关系ID
		grelations[(h,t)].add(r)
		# 存储邻居
		glinks[t].add(h)
		glinks[h].add(t)
		# gold_可以用于查找每个三元组的头实体和尾实体
		gold_heads[(r,t)].add(h)
		gold_tails[(h,r)].add(t)
		# candidate_以关系作为键值，每个键对应的值为与该关系相连的头实体或尾实体
		candidate_heads[r].add(h)
		candidate_tails[r].add(t)
		# 与glinks的存储一样，但glinks的键值包含了所有实体，而这个的键值只有头实体或尾实体
		tail_per_head[h].add(t)
		head_per_tail[t].add(h)
		# 引入规则
		if r in premise:
			conclusion = premise[r]
			for r_, flag in conclusion:
				if flag == 1:
					grelations[(h,t)].add(r_)
					train_data.add((h,r_,t,))
					trfreq[r_]+=1
					gold_heads[(r_,t)].add(h)
					gold_tails[(h,r_)].add(t)
					candidate_heads[r_].add(h)
					candidate_tails[r_].add(t)
				else:
					grelations[(t,h)].add(r_)
					train_data.add((t,r_,h,))
					trfreq[r_]+=1
					gold_heads[(r_,h)].add(t)
					gold_tails[(t,r_)].add(h)
					candidate_heads[r_].add(t)
					candidate_tails[r_].add(h)
					tail_per_head[t].add(h)
					head_per_tail[h].add(t)

	for e in glinks:
		glinks[e] = list(glinks[e])
	for r in candidate_heads:
		candidate_heads[r] = list(candidate_heads[r])
	for r in candidate_tails:		
		candidate_tails[r] = list(candidate_tails[r])
	# 改写值，每个键值对应的为与它有关系的实体数量
	for h in tail_per_head:			
		tail_per_head[h] = len(tail_per_head[h])+0.0
	for t in head_per_tail:			
		head_per_tail[t] = len(head_per_tail[t])+0.0

	# 构造辅助集
	tool.trace('set axiaulity')
	# switch standard setting or OOKB setting
	# 2个实验，标准三元组分类实验和OOKB实验
	if args.train_file==args.auxiliary_file:
		tool.trace('standard setting, use: edges=links')
		gedges = glinks
	else:
		# ookb的实验只修改grelation和gedges?
		tool.trace('OOKB esetting, use: different edges')
		gedges = defaultdict(set)
		for line in tool.read(args.auxiliary_file):
			h,r,t = list(map(int,line.strip().split('\t')))
			grelations[(h,t)].add(r)
			# 引入规则
			if r in premise:
				conclusion = premise[r]
				for r_, flag in conclusion:
					if flag == 1:
						grelations[(h,t)].add(r_)
					else:
						grelations[(t,h)].add(r_)
			gedges[t].add(h)
			gedges[h].add(t)
		for e in gedges:
			gedges[e] = list(gedges[e])

	for (h,t) in grelations:
		grelations[(h,t)] = list(grelations[(h,t)])
		# if len(grelations[(h,t)]) != 1:
		# 	print("len(grelations[(h,t)])",len(grelations[(h,t)]),h,t,grelations[(h,t)])
	print("grelations", len(grelations))
	train_data = list(train_data)
	print("train_data", len(train_data))
	for r in trfreq:
		trfreq[r] = args.train_size/(float(trfreq[r])*len(trfreq))

	# load dev
	# 读取验证集，验证集包含正三元组和负三元组
	tool.trace('load dev')
	dev_data = list()
	for line in open(args.dev_file):
		h,r,t,l = list(map(int,line.strip().split('\t')))
		# 过滤掉验证集中含OOKB实体的三元组(只在做OOKB的实验中会成立？)
		if h not in glinks or t not in glinks: continue
		dev_data.append((h,r,t,l,))
	print('dev size:', len(dev_data))

	# load test
	# 读取测试集，测试集包含正三元组和负三元组
	tool.trace('load test')
	test_data = list()
	for line in open(args.test_file):
		h,r,t,l = list(map(int,line.strip().split('\t')))
		# 过滤掉测试集中含OOKB实体的三元组(做OOKB的实验不是过滤掉不含的吗？)
		# 标准三元组分类的实验中没有OOKB的实体？
		if h not in glinks or t not in glinks: continue
		test_data.append((h,r,t,l,))
	print('test size:', len(test_data))

def generator_train_with_corruption(args):
	# 得到一个小于1的数，以该概率跳过某次的corruption
	skip_rate = args.train_size/float(len(train_data))

	positive,negative=list(),list()

	random.shuffle(train_data)
	for i in range(len(train_data)):
		h,r,t = train_data[i]

		# is_balanced_tr的默认值为false
		if args.is_balanced_tr:
			if random.random()>trfreq[r]: 
				continue
		else:
			if random.random()>skip_rate: 
				continue

		# tph/Z
		head_ratio = 0.5
		if args.is_bernoulli_trick:  #默认为TRUE
			# 设定替换头实体的阈值为:与头实体有关系的尾实体数量÷(与头实体有关系的尾实体数量+与该三元组的尾实体有关系的头实体数量)
			head_ratio = tail_per_head[h]/(tail_per_head[h]+head_per_tail[t])
		# 如果大于阈值，则特换头实体，构建corrupted triplets
		if random.random()>head_ratio:
			cand = random.choice(candidate_heads[r])
			# 选择一个使得(h,r,t)不成立的头实体进行替换
			while cand in gold_heads[(r,t)]:
				cand = random.choice(candidate_heads[r])
			h = cand
		# 否则替换尾实体
		else:
			cand = random.choice(candidate_tails[r])
			while cand in gold_tails[(h,r)]:
				cand = random.choice(candidate_tails[r])
			t = cand
		
		# 将替换前的三元组加入positive，替换后的三元组加入negative
		if len(positive)==0 or len(positive) <= args.batch_size:
			positive.append(train_data[i])
			negative.append((h,r,t))
		# 当len(positive) == args.batch_size时返回positive和negative
		# 然后进入main函数的for循环进行运算
		# 函数保存当前状态，下一次迭代从yield下一句开始
		else:
			yield positive,negative
			positive,negative = [train_data[i]],[(h,r,t)]

	if len(positive)!=0:
		yield positive,negative

#----------------------------------------------------------------------------

def train(args,m,xp,opt):
	Loss,N = list(),0
	# positive中为正三元组，negative中为随机替换头实体或者尾实体的负三元组
	# positive和negative的size均为batch_size
	for positive, negative in generator_train_with_corruption(args):
		loss = m.train(positive,negative,glinks,grelations,gedges,xp)
		loss.backward()  # 调用backward()来实现反向传播的计算
		opt.update()
		Loss.append(float(loss.data)/len(positive))
		N += len(positive)
		del loss
		os.system("pause")
	return sum(Loss),N

def dump_current_scores_of_devtest(args,m,xp):
	for mode in ['dev','test']:
		if mode=='dev': 	current_data = dev_data
		if mode=='test': 	current_data = test_data

		scores, accuracy = list(),list()
		for batch in chunked(current_data, args.test_batch_size):
			with chainer.using_config('train',False), chainer.no_backprop_mode():
				current_score = m.get_scores(batch,glinks,grelations,gedges,xp,mode)
			for v,(h,r,t,l) in zip(current_score.data, batch):
				values = (h,r,t,l,v)
				values = map(str,values)
				values = ','.join(values)
				scores.append(values)
				if v < args.threshold: # 模型判断为正三元组
					if l==1: accuracy.append(1.0)
					else: accuracy.append(0.0)
				else: # 模型判断为负三元组
					if l==1: accuracy.append(0.0)
					else: accuracy.append(1.0)
			del current_score
		tool.trace('\t ',mode,sum(accuracy)/len(accuracy))
		if args.margin_file!='':
			with open(args.margin_file,'a') as wf:
				wf.write(mode+':'+' '.join(scores)+'\n')

def get_sizes(args):
	relation,entity=-1,-1
	for line in open(args.train_file):
		h,r,t = list(map(int, line.strip().split('\t')))
		# 找到关系和实体中最大的标记即为数量，从0开始计数所以要最后+1
		relation = max(relation, r)
		entity = max(entity, h, t)
	return relation+1, entity+1

import sys
def main(args):
	# 读入规则
	rules_addition()
	# 初始化所有数据集
	initilize_dataset()
	# 根据文件内容修改关系和实体的默认值
	args.rel_size,args.entity_size = get_sizes(args)
	print('relation size:',args.rel_size,'entity size:',args.entity_size)

	xp = Backend(args)  #返回一个可调用的对象
	m = get_model(args)  # return A0(args)
	# Setup an optimizer
	# 设置训练时用的优化方法，默认为Adam
	opt = get_opt(args)  # return optimizers.Adam()
	# setup()方法只是为优化器提供一个link
	opt.setup(m)
	for epoch in range(args.epoch_size):
		opt.alpha = args.beta0/(1.0+args.beta1*epoch)
		trLoss,Ntr = train(args,m,xp,opt)  #对应到main.py的train
		tool.trace('epoch:',epoch,'tr Loss:',tool.dress(trLoss),Ntr)
		dump_current_scores_of_devtest(args,m,xp)


#----------------------------------------------------------------------------
"""
	-tF dataset/data/Freebase13/train \
	-dF dataset/data/Freebase13/train \
	-eF dataset/data/Freebase13/test \
"""
from argparse import ArgumentParser
def argument():
	p = ArgumentParser()

	# GPU	
	p.add_argument('--use_gpu',     '-g',   default=False,  action='store_true')
	p.add_argument('--gpu_device',  '-gd',  default=0,      type=int)

	# trian, dev, test, and other filds
	p.add_argument('--train_file',      '-tF',  default='datasets/standard/WordNet11/serialized/train')
	p.add_argument('--dev_file',        '-vF',  default='datasets/standard/WordNet11/serialized/dev')
	p.add_argument('--test_file',       '-eF',  default='datasets/standard/WordNet11/serialized/test')
	p.add_argument('--auxiliary_file',  '-aF',  default='datasets/standard/WordNet11/serialized/train')
	p.add_argument('--rules_file',  '-rF',  default='one2one_rules.txt')
	# dirs
	p.add_argument('--param_dir',       '-pD',  default='')
	p.add_argument('--margin_file',     '-mF',  default='margin_file.txt')

	# entity and relation sizes
	p.add_argument('--rel_size',  	'-Rs',      default=11,			type=int)
	p.add_argument('--entity_size', '-Es',      default=38194,		type=int)

	# model parameters (neural network)
	p.add_argument('--nn_model',		'-nn',  default='A0')
	p.add_argument('--activate',		'-af',  default='relu')
	p.add_argument('--pooling_method',	'-pM',  default='max')
	p.add_argument('--dim',         '-D',       default=200,    type=int)
	p.add_argument('--order',       '-O',       default=1,      type=int)
	p.add_argument('--threshold',   '-T',       default=300.0,  type=float)
	p.add_argument('--layerR' ,		'-Lr',      default=1,      type=int)

	# objective function
	p.add_argument('--objective_function',   '-obj',    default='absolute')

	# dropout rates
	p.add_argument('--dropout_block','-dBR',     default=0.0,  type=float)
	p.add_argument('--dropout_decay','-dDR',     default=0.0,   type=float)
	p.add_argument('--dropout_embed','-dER',     default=0.0,   type=float)

	# model flags
	p.add_argument('--is_residual',     '-nR',   default=False,   action='store_true')
	p.add_argument('--is_batchnorm',    '-nBN',  default=True,   action='store_false')
	p.add_argument('--is_embed',      	'-nE',   default=True,   action='store_false')
	p.add_argument('--is_known',    	'-iK',   default=False,   action='store_true')
	p.add_argument('--is_bound_wr',		'-iRB',  default=True,   action='store_false')

	# parameters for negative sampling (corruption)
	p.add_argument('--is_balanced_tr',    '-iBtr',   default=False,   action='store_true')
	p.add_argument('--is_balanced_dev',   '-nBde',   default=True,   action='store_false')
	p.add_argument('--is_bernoulli_trick', '-iBeT',  default=True,   action='store_false')

	# sizes
	p.add_argument('--train_size',  	'-trS',  default=100000,       type=int)
	p.add_argument('--batch_size',		'-bS',  default=5000,        type=int)
	p.add_argument('--test_batch_size', '-tbS',  default=20000,        type=int)
	p.add_argument('--sample_size',		'-sS',  default=64,        type=int)
	p.add_argument('--pool_size',		'-pS',  default=128*5,      type=int)
	p.add_argument('--epoch_size',		'-eS',  default=1000,       type=int)

	# optimization
	p.add_argument('--opt_model',   "-Op",  default="Adam")
	p.add_argument('--alpha0',      "-a0",  default=0,      type=float)
	p.add_argument('--alpha1',      "-a1",  default=0,      type=float)
	p.add_argument('--alpha2',      "-a2",  default=0,      type=float)
	p.add_argument('--alpha3',      "-a3",  default=0,      type=float)
	p.add_argument('--beta0',       "-b0",  default=0.01,   type=float)
	p.add_argument('--beta1',       "-b1",  default=0.0001,  type=float)

	# seed to control generaing random variables
	p.add_argument('--seed',        '-seed',default=0,      type=int)

	args = p.parse_args()
	return args

import random
if __name__ == '__main__':
	args = argument()
	print(args)
	print(' '.join(sys.argv))
	main(args)

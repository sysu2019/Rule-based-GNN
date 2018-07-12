#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

from collections import defaultdict
import sys,random
import numpy as np
import os

global en
en = defaultdict(set)

class Module(chainer.Chain):
	def __init__(self, dim, dropout_rate, activate, isR, isBN):
		super(Module, self).__init__(
			# L.Linear(in_size,out_size) x2z：输入向量维度为dim,输出向量维度也为dim
			x2z	=L.Linear(dim,dim),
			# 线性或卷积函数输出上的批量标准化层,第一个参数是size
			bn	=L.BatchNormalization(dim)
		)
		self.dropout_rate=dropout_rate  # 0.05
		self.activate = activate  # tanh
		self.is_residual = isR  # False
		self.is_batchnorm = isBN  # False
	
	def __call__(self, x):
		if self.dropout_rate!=0:
			# F.dropout以概率比率随机丢弃输入元素，并按比例因子1/(1-比例)缩放其余元素，返回x
			x = F.dropout(x,ratio=self.dropout_rate)
		# z的维度和x一样
		z = self.x2z(x)
		# 批量归一化
		if self.is_batchnorm:
			z = self.bn(z)
		# 转换函数：用于修改相邻节点的向量以反映当前节点和邻居的关系
		if self.activate=='tanh': z = F.tanh(z)
		if self.activate=='relu': z = F.relu(z)
		
		if self.is_residual:	return z+x
		else: return z

class Block(chainer.Chain):
	def __init__(self, dim, dropout_rate, activate, layer, isR, isBN):
		super(Block, self).__init__()
		# 每层layer的links
		links = [('m{}'.format(i), Module(dim,dropout_rate,activate, isR, isBN)) for i in range(layer)]
		for link in links:
			# *link是以tuple的形式传递无名的link，表示接收的参数作为元组来处理
			# self.forward[i][0] = mi
			self.add_link(*link)
		self.forward = links
	def __call__(self,x):
		for name, _ in self.forward:
			x = getattr(self,name)(x)
		return x

class Tunnel(chainer.Chain):
	def __init__(self, dim, dropout_rate, activate, layer, isR, isBN, relation_size, pooling_method):
		super(Tunnel, self).__init__()
		linksH = [('h{}'.format(i), Block(dim,dropout_rate,activate,layer,isR,isBN)) for i in range(relation_size)]
		for link in linksH:
			self.add_link(*link)
		self.forwardH = linksH
		linksT = [('t{}'.format(i), Block(dim,dropout_rate,activate,layer,isR,isBN)) for i in range(relation_size)]
		for link in linksT:
			self.add_link(*link)
		self.forwardT = linksT
		self.pooling_method = pooling_method
		self.layer = layer  # 默认值1

	# result(xs)为relu后的返回值
	# assign(neighbor)为(邻居编号v,与该邻居有关的实体编号i的list)
	def maxpooling(self,xs,neighbor):
		sources = defaultdict(list)
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		# sources:键值为实体编号i,值为与实体有关系的所有实体经转换函数后的embedding
		# 将sources根据实体集的编号排序
		# 最后得到的len(result)=len(entities)
		result = []
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: 
				result.append(xxs[0])
			else:
				x = F.concat(xxs,axis=0)					# -> (b,d)
				x = F.swapaxes(x,0,1)						# -> (d,b)
				x = F.maxout(x,len(xxs))					# -> (d,1)
				x = F.swapaxes(x,0,1)						# -> (1,d)
				result.append(x)
		return result

	def averagepooling(self,xs,neighbor):
		sources = defaultdict(list)
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		result = []
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: 
				result.append(xxs[0])
			else:			
				result.append(sum(xxs)/len(xxs))
		return result

	def sumpooling(self,xs,neighbor):
		sources = defaultdict(list)
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		result = []
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: result.append(xxs[0])
			else:			result.append(sum(xxs))
		return result

	def easy_case(self,x,neighbor_entities,neighbor_dict,assign,entities,relations):

		x = F.split_axis(x,len(neighbor_dict),axis=0)

		assignR = dict()
		bundle = defaultdict(list)
		for v,k in enumerate(neighbor_entities):
			for i in assign[v]:
				e = entities[i]
				if (e,k) in relations:	
					for r in relations[(e,k)]:
						r =  relations[(e,k)]*2
						assignR[(r,len(bundle[r]))] = v
						bundle[r].append(x[v])
				else:
					for r in relations[(e,k)]:
						r =  relations[(k,e)]*2+1
						assignR[(r,len(bundle[r]))] = v
						bundle[r].append(x[v])

		result = [0 for i in range(len(neighbor_dict))]
		for r in bundle:
			rx = bundle[r]
			if len(rx)==1:	result[assignR[(r,0)]] = rx[0]
			else:
				for i,x in enumerate(rx):
					result[assignR[(r,i)]] = x

		if self.pooling_method=='max':
			result = self.maxpooling(result,assign)
		if self.pooling_method=='avg':
			result = self.averagepooling(result,assign)
		if self.pooling_method=='sum':
			result = self.sumpooling(result,assign)
		result = F.concat(result,axis=0)
		return result



	"""
	# neighbor_entities=[(k,v)]
	# (e,k) in links
	# e = entities[i]
	# i in assing[v]
	"""
	"""
	source entityから出てるedgeが無い
	"""
	# neighbor_entities为邻居ID k
	# x为neighbor_entities对应的embedding
	# neighbor_dict为(邻居ID k，对应编号 v)
	# assign为(邻居编号v, 与该邻居有关的实体编号i的list)
	# len(x)=len(neighbor_dict)=len(neighbor_entities)=len(assign)
	# 邻居数量为27621，调用此函数后结果result为每个实体更新的embedding，数量为len(entities)=11337
	def __call__(self,x,neighbor_entities,neighbor_dict,assign,entities,relations):
		if self.layer==0:
			return self.easy_case(x,neighbor_entities,neighbor_dict,assign,entities,relations)

		if len(neighbor_dict)==1:
			x=[x]
		else:
			# print("x.shape", x.shape) #(27621L,200L)
			# 把每个embedding分割成二维数组, len(x)=len(neighbor_dict), x[0]=variable([[2...0]])
			x = F.split_axis(x, len(neighbor_dict),axis=0)

		assignR = dict()
		bundle = defaultdict(list)

		for v,k in enumerate(neighbor_entities):
			for i in assign[v]:
				# 得到编号为i的实体的实体ID
				e = entities[i]
				# 凭借r是奇数还是偶数区分(k,r,e)和(e,r,k)两种情况
				# relations[(e,k)]，k为尾实体的情况，返回关系ID，每种关系ID对应唯一的r
				if (e,k) in relations:
					for r_ in relations[(e,k)]:
						r =  r_ * 2
						# 用len(bundle[r])来编号，bundle[r]的值的序号对应assignR[(r,l)]中的l
						assignR[(r, len(bundle[r]))] = v
						# 编号为v的邻居实体向量的embedding
						bundle[r].append(x[v])
				# k为头实体的情况
				elif (k,e) in relations:
					for r_ in relations[(k,e)]:			
						r = r_ * 2 + 1
						assignR[(r, len(bundle[r]))] = v
						bundle[r].append(x[v])

		result = [0 for i in range(len(neighbor_dict))]
		for r in bundle:
			rx = bundle[r]  # list, embedding
			# 即包含该关系的三元组只有一组
			if len(rx)==1:
				rx=rx[0]
				# 得到<e,r,k>forwardH 表示头实体是邻居的情况的传播
				if r%2==0:	rx = getattr(self, self.forwardH[r//2][0])(rx)
				# 得到<h,r,e>forwardT 表示尾实体是邻居的情况的传播
				else:		rx = getattr(self, self.forwardT[r//2][0])(rx)
				result[assignR[(r,0)]] = rx
			else:
				size = len(rx)
				rx = F.concat(rx,axis=0)  # 按行拼成一个矩阵
				if r%2==0:	rx = getattr(self,self.forwardH[r//2][0])(rx)
				else:		rx = getattr(self,self.forwardT[r//2][0])(rx)
				rx = F.split_axis(rx,size,axis=0)
				# x的值为经过转换函数之后的embedding,shape=(1L,200L)
				for i,x in enumerate(rx):
					result[assignR[(r,i)]] = x

		if self.pooling_method=='max':
			result = self.maxpooling(result,assign)
		if self.pooling_method=='avg':
			result = self.averagepooling(result,assign)
		if self.pooling_method=='sum':
			result = self.sumpooling(result,assign)
		result = F.concat(result,axis=0)

		return result



class Model(chainer.Chain):
	def __init__(self, args):
		super(Model, self).__init__(
			# 类似于初始化variable
			# 之后通过返回值可以得到更新
			embedE	= L.EmbedID(args.entity_size,args.dim),
			embedR	= L.EmbedID(args.rel_size,args.dim),
		)
		linksB = [('b{}'.format(i), Tunnel(args.dim,args.dropout_block,args.activate,args.layerR,args.is_residual,args.is_batchnorm, args.rel_size, args.pooling_method)) for i in range(args.order)]
		for link in linksB:
			self.add_link(*link)
		self.forwardB = linksB  #B for Both  表示包含了 forwardH 和 forwaT两种情况

		self.sample_size = args.sample_size
		self.dropout_embed = args.dropout_embed
		self.dropout_decay = args.dropout_decay
		self.depth = args.order
		self.is_embed = args.is_embed
		self.is_known = args.is_known
		self.threshold = args.threshold
		self.objective_function = args.objective_function
		self.is_bound_wr = args.is_bound_wr
		if args.use_gpu: self.to_gpu()


	def get_context(self,entities,links,relations,edges,order,xp):
		# xp = Backend(args)
		# 调用xp.array后return Variable(self.lib.array(entities, dtype=self.lib.int32)
		# entities为一个list，调用embedE后返回这个list里所有entity的embedding
		if self.depth==order:
			return self.embedE(xp.array(entities,'i'))

		assign = defaultdict(list)
		neighbor_dict = defaultdict(int)

		# 获得与实体集entities有关系的所有实体
		for i,e in enumerate(entities):
			"""
			(not self.is_known)
				unknown setting
			(not is_train)
				in test time
			order==0
				in first connection
			"""
			# glinks(links)的键值包含了训练集所有实体ID，值为与该实体有关系的实体
			if e in links:
				if len(links[e])<=self.sample_size:	
					nn = links[e]
				else:
					nn = random.sample(links[e], self.sample_size)
				if len(nn)==0:
					print('something wrong @ modelS')
					print('entity not in links',e,self.is_known,order)
					sys.exit(1)
			# 如果不在glinks中，则为OOKB的实体
			else:
				# gedges(edges)同glinks一样，键值包含所有实体，值为与该实体有关的实体
				if len(edges[e])<=self.sample_size:	nn = edges[e]
				else:		nn = random.sample(edges[e],self.sample_size)
				if len(nn)==0:
					print('something wrong @ modelS')
					print('entity not in edges',e,self.is_known,order)
					sys.exit(1)

			# nn为与该实体e有关系的实体集，size为sample_size
			for k in nn:
				if k not in neighbor_dict:
					# 给neighbor编号，实体k在neighbor_dict的编号为v
					neighbor_dict[k] = len(neighbor_dict)	# (k,v)
				# 实体k可能与多个e有关系，实体k的编号为v
				# assign的键值为实体k的编号，值为与k有关的实体e(编号为i)的list
				assign[neighbor_dict[k]].append(i)

		neighbor = []
		# 按v的大小从小到大进行排序
		for k,v in sorted(neighbor_dict.items(), key=lambda x:x[1]):
			neighbor.append(k)
		# 此处调用函数本身有depth==order，因此返回neighbor的embedding
		x = self.get_context(neighbor,links,relations,edges,order+1,xp)
		# 返回后会更新embedding，即self.embedE
		x = getattr(self, self.forwardB[order][0])(x,neighbor,neighbor_dict,assign,entities,relations)
		return x

	# main函数中调用m.train(positive,negative,glinks,grelations,gedges,xp)
	def train(self,positive,negative,links,relations,edges,xp):
		# 每次调用之前，需要对梯度进行清零操作
		self.cleargrads()

		entities= set()
		for h,r,t in positive:
			entities.add(h)
			entities.add(t)
		for h,r,t in negative:
			entities.add(h)
			entities.add(t)
		# 获取正负三元组的所有实体
		entities = list(entities)

		# x为entities经过传播模型所得的embedding
		x = self.get_context(entities,links,relations,edges,0,xp)
		x = F.split_axis(x,len(entities),axis=0)
		edict = dict()
		for e,x in zip(entities,x):
			edict[e]=x

		pos,rels = [],[]
		for h,r,t in positive:
			rels.append(r)
			pos.append(edict[h]-edict[t])
		pos = F.concat(pos,axis=0)
		xr = self.embedR(xp.array(rels,'i'))  # 返回r的embedding
		if self.is_bound_wr:	xr = F.tanh(xr)  # 对之前的embedR会有更新
		pos = F.batch_l2_norm_squared(pos+xr)  # (h+r-t)的L2正则

		neg,rels = [],[]
		for h,r,t in negative:
			rels.append(r)
			neg.append(edict[h]-edict[t])
		neg = F.concat(neg,axis=0)
		xr = self.embedR(xp.array(rels,'i'))
		if self.is_bound_wr:	xr = F.tanh(xr)
		neg = F.batch_l2_norm_squared(neg+xr)

		# 输出模型
		if self.objective_function=='relative': 
			return sum(F.relu(self.threshold+pos-neg)) # 返回Loss
		if self.objective_function=='absolute': 
			return sum(pos+F.relu(self.threshold-neg))


	def get_scores(self,candidates,links,relations,edges,xp,mode):
		entities = set()
		for h,r,t,l in candidates:
			entities.add(h)
			entities.add(t)
		entities = list(entities)
		# 返回entities经传播模型后的embedding
		xe = self.get_context(entities,links,relations,edges,0,xp)
		xe = F.split_axis(xe,len(entities),axis=0)
		edict = dict()
		for e,x in zip(entities,xe):
			edict[e]=x
		diffs,rels = [],[]
		for h,r,t,l in candidates:
			rels.append(r)
			diffs.append(edict[h]-edict[t])
		diffs = F.concat(diffs,axis=0)
		xr = self.embedR(xp.array(rels,'i'))
		if self.is_bound_wr:	xr = F.tanh(xr)
		scores = F.batch_l2_norm_squared(diffs+xr)
		return scores

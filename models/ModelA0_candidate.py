#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

from collections import defaultdict
import sys,random
import numpy as np
import os

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
	def __init__(self, dim, dropout_rate, activate, layer, isR, isBN, relation_size, pooling_method, AModule_size):
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

		#self.forwardAA=Attention(AModule_size,dim)

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
				x=xxs[0]
				# x=self.forwardAA(x,xxs) # attention
				result.append(x)
			else:
				x = F.concat(xxs,axis=0)					# -> (b,d)
				x = F.swapaxes(x,0,1)						# -> (d,b)
				x = F.maxout(x,len(xxs))					# -> (d,1)
				x = F.swapaxes(x,0,1)						# -> (1,d)
				# x=self.forwardAA(x,xxs)  # attention
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
				x=xxs[0]
				# x=self.forwardAA(x,xxs) # attention
				result.append(x)
			else:			
				result.append(sum(xxs)/len(xxs))
				# # x为pooling后实体embedding,xxs为邻居embedding做relu后的结果
				# x=sum(xxs)/len(xxs) # x.shape=(1,200)
				# x=self.forwardAA(x,xxs) # (x, xxs)=(entity, neighbors)
				# result.append(x)
		return result

	def sumpooling(self,xs,neighbor):
		sources = defaultdict(list)
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		result = []
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: 
				# result.append(xxs[0])
				x=xxs[0]
				x=self.forwardAA(x,xxs) # attention
				result.append(x)
			else:			
				# result.append(sum(xxs))
				x=sum(xxs)
				x=self.forwardAA(x,xxs) # attention
				result.append(x)
		return result

	# def easy_case(self,n_embed,r_embed,neighbor,rels,candidates_dict,assign,entities,relations):

	# 	n_embed = F.split_axis(n_embed, len(candidates_dict),axis=0)
	# 	r_embed = F.split_axis(r_embed, len(candidates_dict),axis=0)

	# 	assignR = dict()
	# 	bundle = defaultdict(list)
	# 	for v,k in enumerate(candidates_dict):
	# 		for i in assign[v]:
	# 			e = entities[i]
	# 			if (e,k) in relations:	
	# 				for r in relations[(e,k)]:
	# 					r =  relations[(e,k)]*2
	# 					assignR[(r,len(bundle[r]))] = v
	# 					bundle[r].append(x[v])
	# 			else:
	# 				for r in relations[(e,k)]:
	# 					r =  relations[(k,e)]*2+1
	# 					assignR[(r,len(bundle[r]))] = v
	# 					bundle[r].append(x[v])

	# 	result = [0 for i in range(len(neighbor_dict))]
	# 	for r in bundle:
	# 		rx = bundle[r]
	# 		if len(rx)==1:	result[assignR[(r,0)]] = rx[0]
	# 		else:
	# 			for i,x in enumerate(rx):
	# 				result[assignR[(r,i)]] = x

	# 	if self.pooling_method=='max':
	# 		result = self.maxpooling(result,assign)
	# 	if self.pooling_method=='avg':
	# 		result = self.averagepooling(result,assign)
	# 	if self.pooling_method=='sum':
	# 		result = self.sumpooling(result,assign)
	# 	result = F.concat(result,axis=0)
	# 	return result



	"""
	# neighbor_entities=[(k,v)]
	# (e,k) in links
	# e = entities[i]
	# i in assing[v]
	"""
	"""
	source entityから出てるedgeが無い
	"""
	# neighbor为candidate中的邻居ID
	# rels为candidate中的关系ID
	# n_embed为neighbor对应的embedding, list
	# r_embed为rels对应的embedding, list
	# candidates_dict为((r,e,flag)，对应编号v)
	# assign为(candidate编号v, 与candidate有关的实体编号i的list)
	# 调用此函数后结果result为每个实体更新的embedding，数量为len(entities)=11337
	def __call__(self,n_embed,r_embed,neighbor,rels,candidates_dict,assign,entities,relations):
		# if self.layer==0:
		# 	return self.easy_case(n_embed,r_embed,neighbor,rels,candidates_dict,assign,entities,relations)
		assert len(n_embed) == len(r_embed) and len(neighbor) == len(candidates_dict)
		if len(candidates_dict)==1:
			n_embed=[n_embed]
			r_embed=[r_embed]
		else:
			# print("x.shape", x.shape) #(27621L,200L)
			# 把每个embedding分割成二维数组, len(n_embed)=len(candidates_dict), x[0]=variable([[2...0]])
			n_embed = F.split_axis(n_embed, len(candidates_dict),axis=0)
			r_embed = F.split_axis(r_embed, len(candidates_dict),axis=0)

		assignR = dict()  # assignR=((r,bundle[r]序号l), candidate序号v)
		bundle = defaultdict(list)  # bundle=(r,embed)

		for v,k in enumerate(candidates_dict):
			for i in assign[v]:
				# 得到编号为i的实体的实体ID
				e = entities[i]
				can_r, can_e, flag = k
				if flag == 1:
					# flag = 1即(e,r,t),candidate_of_e = t-r
					# assignR和bundle中凭借r是奇数还是偶数区分(h,r,e)和(e,r,t)两种情况
					r = can_r * 2
					assignR[(r, len(bundle[r]))] = v
					bundle[r].append(n_embed[neighbor.index(can_e)] - r_embed[rels.index(can_r)])
				elif flag == 0:
					# flag = 0即(h,r,e),candidate_of_e = h+r
					r = can_r * 2 + 1
					assignR[(r, len(bundle[r]))] = v
					bundle[r].append(n_embed[neighbor.index(can_e)] + r_embed[rels.index(can_r)])
				
		result = [0 for i in range(len(candidates_dict))]
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
				for i,embed in enumerate(rx):
					result[assignR[(r,i)]] = embed

		# len(result) = len(assign)
		if self.pooling_method=='max':
			result = self.maxpooling(result,assign)
		if self.pooling_method=='avg':
			result = self.averagepooling(result,assign)
		if self.pooling_method=='sum':
			result = self.sumpooling(result,assign)
		result = F.concat(result,axis=0)
		assert len(result) == len(entities)
		return result

# # 得到f(ei, ej)
# class AModule(chainer.Chain):
# 	def __init__(self, dim, activate):
# 		super(AModule, self).__init__(
# 			es2e=L.Linear(dim*2, 1)
# 		)
# 		self.dim = dim
# 		self.activate = activate
# 	def __call__(self, e1, e2): # e1 e2为embedding
# 		e12=F.concat((e1,e2),axis=1)
# 		eij=self.es2e(e12)
# 		if self.activate=='tanh': eij = F.tanh(eij)
# 		return eij

# class Attention(chainer.Chain):
# 	def __init__(self, AModule_size, dim):
# 		super(Attention,self).__init__()
# 		# AModule的个数为超参数
# 		# 设为1即一个结果的attention,设为3即三个结果最后取平均
# 		linksA=[('a{}'.format(i),AModule(dim, 'tanh')) for i in range(AModule_size)]
# 		for link in linksA:
# 			self.add_link(*link)
# 		self.forwardA=linksA
# 		self.AModule_size=AModule_size

# 	def __call__(self,entity,neighbors):

# 		es=defaultdict(list)
# 		for j in range(self.AModule_size):
# 			name=self.forwardA[j][0]

# 			es[j]=[0 for i in range(len(neighbors))]
# 			for n in range(len(neighbors)):
# 				es[j][n]=getattr(self,name)(entity,neighbors[n])  # variable([[-0.x]])
# 			es[j][0]=F.softmax(F.concat(es[j]),axis=1) # shape=(1, len(neighbors))

# 		for j in range(self.AModule_size-1):
# 			es[0][0]=es[0][0]+es[j+1][0]
# 		es[0][0]=es[0][0]/self.AModule_size

# 		aentity=self.attentionpooling(neighbors,es[0][0])
# 		return aentity
		
# 	def attentionpooling(self,neighbors,es):
# 		# es.shape=(1, len(neighbors))
# 		# neighbors.shape=(len(neighbors), 1, dim)
# 		aentity=0
# 		for i in range(len(neighbors)):
# 			exm=[es[0][i].reshape(1,1) for j in range(neighbors[i].shape[1])]  # shape=(dim,1,1)
# 			exm=F.concat(exm,axis=1)  # exm.shape=(1,dim)
# 			aentity+=neighbors[i]*exm  # neighbors[i].shape=(1,dim)
# 			# aentity.shape=(1,dim)
# 		return aentity

class Model(chainer.Chain):
	def __init__(self, args):
		super(Model, self).__init__(
			# 类似于初始化variable
			# 之后通过返回值可以得到更新
			embedE	= L.EmbedID(args.entity_size,args.dim),
			embedR	= L.EmbedID(args.rel_size,args.dim),
		)
		linksB = [('b{}'.format(i), Tunnel(args.dim,args.dropout_block,args.activate,args.layerR,args.is_residual,args.is_batchnorm, args.rel_size, args.pooling_method, args.AModule_size)) for i in range(args.order)]
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
		self.AModule_size = args.AModule_size

		if args.use_gpu: self.to_gpu()


	def get_context(self,entities,erels,links,relations,edges,order,xp):
		# xp = Backend(args)
		# 调用xp.array后return Variable(self.lib.array(entities, dtype=self.lib.int32)
		# entities为一个list，调用embedE后返回这个list里所有entity的embedding
		if self.depth==order:
			e_embed = self.embedE(xp.array(entities,'i'))
			r_embed = self.embedR(xp.array(erels,'i'))
			return e_embed,r_embed

		assign = defaultdict(list)
		candidates_dict = defaultdict(int)

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
			# glinks(links)的键值包含了训练集所有实体ID,值为(r,e,flag)
			# gedges(edges)的键值包含了辅助集所有实体ID,值同glinks一样
			if e in links:
				cc = links[e]
			# 如果不在glinks中，则为OOKB的实体
			else:
				cc = edges[e]
			if len(cc)==0:
				print('something wrong @ modelS')
				print('entity not in edges',e,self.is_known,order)
				sys.exit(1)

			# cc为与该实体e有关系的candidates
			for (a,b,flag) in cc:
				if (a,b,flag) not in candidates_dict:
					# 给candidates编号,(a,b,flag)在candidates_dict的编号为v
					candidates_dict[(a,b,flag)] = len(candidates_dict)	# (键值k,编号v)
				# k可能与多个e有关系，k的编号为v
				# assign的键值为k的编号，值为与k有关的实体e(编号为i)的list
				assign[candidates_dict[(a,b,flag)]].append(i)
		assert len(assign) == len(candidates_dict)

		neighbor = []  # 存储candidate的实体ID
		rels = []  # 存储candidate的关系ID
		# 按v的大小从小到大进行排序
		for k,v in sorted(candidates_dict.items(), key=lambda x:x[1]):
			neighbor.append(k[1])
			rels.append(k[0])
		assert len(neighbor) == len(rels)

		# 此处调用函数本身有depth==order，因此返回neighbor和relations的embedding
		n_embed,r_embed = self.get_context(neighbor,rels,links,relations,edges,order+1,xp)
		# 返回后会更新embedding，即self.embedE和self.embedR
		x = getattr(self, self.forwardB[order][0])(n_embed,r_embed,neighbor,rels,candidates_dict,assign,entities,relations)
		assert len(x) == len(entities)
		return x

	# main函数中调用m.train(positive,negative,glinks,grelations,gedges,xp)
	def train(self,positive,negative,links,relations,edges,xp):
		# 每次调用之前，需要对梯度进行清零操作
		self.cleargrads()

		entities= set()
		erels = set()
		for h,r,t in positive:
			entities.add(h)
			entities.add(t)
			erels.add(r)
		for h,r,t in negative:
			entities.add(h)
			entities.add(t)
			erels.add(r)
		# 获取正负三元组的所有实体
		entities = list(entities)
		erels = list(erels)

		# x为entities经过传播模型所得的embedding
		x = self.get_context(entities,erels,links,relations,edges,0,xp)
		x = F.split_axis(x,len(entities),axis=0) # x.shape(len(entities), 1, dim)

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
		erels = set()
		for h,r,t,l in candidates:
			entities.add(h)
			entities.add(t)
			erels.add(r)
		entities = list(entities)
		erels = list(erels)
		# 返回entities经传播模型后的embedding
		xe = self.get_context(entities,erels,links,relations,edges,0,xp)
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

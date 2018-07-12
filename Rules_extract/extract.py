import os
import sys


def get_name():

	file = open('./vocab/entity','r')

	id2entity = {}

	for item in file:
		item = item.strip().split('\t')
		id2entity[item[1]] = item[0]

	file.close()

	file = open('./vocab/relation','r')

	id2relation = {}

	for item in file:
		item = item.strip().split('\t')
		id2relation[item[1]] = item[0]

	file.close()

	return id2entity,id2relation

def num_to_triple(path,id2entity,id2relation):

	file = open(path+'aux_','r')
	triple_file = open(path+path[2:-1]+'triples.tsv','w')
	for item in file:
		item = item.strip().split('\t')
		triple_file.write('<'+id2entity[item[0]]+'>'+'\t'+'<'+id2relation[item[1]]+'>'+'\t'+'<'+id2entity[item[2]]+'>'+'\n')

	file.close()
	triple_file.close()


if __name__ == '__main__':
	path = ['./both-1000/','./both-3000/','./both-5000/','./head-1000/','./head-3000/','./head-5000/','./tail-1000/','./tail-3000/','./tail-5000/']
	id2entity,id2relation = get_name()
	for item in path:
		num_to_triple(item,id2entity,id2relation)



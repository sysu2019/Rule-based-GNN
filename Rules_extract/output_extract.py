import codecs
import os
import string
from collections import defaultdict

# 读取文件
def read(file_path):
	with codecs.open(file_path, 'r', 'utf-8', errors='ignore') as read_f:
		for line in read_f:
			yield line.strip()

# 将实体真实数据对应到ID
def get_name():
	file = open('relation','r')
	relation2id = {}
	for item in file:
		item = item.strip().split('\t')
		relation2id[item[0]] = item[1]
	file.close()
	return relation2id

# 将一对一的规则写入文件,格式为(r, r1, flag)
# r1表示推出来的规则,flag表示该规则前后实体是否一致
# 如果一致则flag=1,否则需要在读入时交换头尾实体
def write_one2one(one2one, path):
	rules_file = open(path+'.txt','w')
	for item in one2one:
		r = list(one2one[item])
		for r1,flag in r:
			rules_file.write(str(item)+'\t'+str(r1)+'\t'+str(flag)+'\n')
	rules_file.close()

if __name__ == '__main__':
	one2one = defaultdict(set)
	multiple2one = defaultdict(set)
	relation2id = get_name()
	for line in read("rules.txt"):
		if line[0] == '?':
			rules = list(line.strip().split('\t'))
			# 置信度>0.8才视作可行的规则
			if rules[3] < 0.8:
				continue
			rules = list(rules[0].strip().split())
			if len(rules) > 7:
				continue
			# len(rules)=7表示该规则为一对一的规则
			elif len(rules) == 7:
				r1 = rules[1].strip('<').strip('>')
				r2 = rules[5].strip('<').strip('>')
				r1 = relation2id[r1]
				r2 = relation2id[r2]
				flag = (rules[0] == rules[4])
				one2one[int(r1)].add((int(r2),int(flag)))
	write_one2one(one2one, 'one2one_rules')
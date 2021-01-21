
import pandas as pd
import numpy as np
import re
import itertools
import pickle 
import json 
#对

file = pd.read_csv('/home/tv/Downloads/query_result_desc.csv',)

tags = file['tags']

#cate_tag_dict = file.loc[:,['category','tags']].set_index().T.to_dict()
cate_tag_dict = file.loc[:,['category','tags']].T.to_dict()



#cate_tag_dict_new =  v if v['tags']=='nan'   for  k,v 
for k,v in cate_tag_dict.items():
    if 'nan' in str(v['tags']):
        v['tags']=v['category']
        cate_tag_dict[k]=v
    #tmp = list(re.split('[\：|\~|\;|\，|\、|\t|\s]',v['tags']))
    cate_tag_dict[k]['tags'] = re.split('[\：|\~|\;|\，|\、|\t|\s]',v['tags'])


tags_raw = [value['tags'] for value in cate_tag_dict.values()]

# [\~|\·|\;|\，|\、|\t|\s]
tags_dicc =list( map(lambda x: re.split('[\：|\~|\;|\，|\、|\t|\s]',x), tags_raw))


tags_ll = list(itertools.chain.from_iterable(list(tags_dicc)))
tags_l = list(set(tags_ll))
len(tags_l)


cate_raw = [value['category'] for value in cate_tag_dict.values()]
cate_l = list(set(cate_raw))

length = len(tags_l)+len(cate_l)

index = [x for x in range(length)]

cate_tag_dict = dict(zip(cate_l+tags_l,index))

cate_tag_dict['UNK']=length

output = open("cate_tag_dict.pkl",'wb')
pickle.dump(cate_tag_dict,output)

with open('cate_tag_dict.json','w') as ff:
    json.dump(cate_tag_dict,ff,ensure_ascii=False)


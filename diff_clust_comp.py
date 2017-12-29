from numpy import *
import numpy as np
import re
from collections import Counter
from difflib import SequenceMatcher
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics 
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
import timeit

start=timeit.default_timer()
with open ("SMSSpam", "r") as myfile:
 data=myfile.readlines()


X=[]
data1=[]
data2=[]

for i in range(0,len(data)):
 X=data[i].split('\t')
 data1.append(X[0])
 data2.append(X[1])

token_word=np.str(data2)
pattern = re.compile('[^A-Za-z0-9 -]')
new_string = pattern.sub('',token_word)


new_string1=new_string.split(' ')
term_freq=Counter(new_string1)
terms=unique(new_string1)
freq=[]

for i in range(0,len(terms)):
 freq.append(term_freq[terms[i]])


freq_avg=average(freq)
indx_pop_terms=np.where(freq>freq_avg)
freq1=np.array(freq)

def similar(a, b):
 return SequenceMatcher(None, a, b).ratio()


indx_spam=[i for i, j in enumerate(data1) if j == 'spam']
indx_ham=[i for i, j in enumerate(data1) if j == 'ham']

cent1=raw_input('Choose initial cluster point1 for ham\t')
cent2=raw_input('Choose initial cluster point2 for spam\t')


cent1=np.int(cent1)
cent2=np.int(cent2)

clust1=indx_ham[cent1]
clust2=indx_spam[cent2]
score1=[]
score2=[]
for i in range(0,len(data2)):
 score1.append(similar(data2[i],data2[clust1]))
 score2.append(similar(data2[i],data2[clust2]))
 
score_clust1=np.array(score1)
score_clust2=np.array(score2)

sim_matrix=np.vstack((score_clust1,score_clust2))
sim_matrix=np.transpose(sim_matrix)


kmeans=KMeans(n_clusters=2,max_iter=900,n_init=10,random_state=0,tol= 0.000005).fit(sim_matrix)
final=kmeans.labels_
#final=sim_matrix.argmin(axis=1)
print "*********** 0 is for ham\n 1 is for spam********";

labels,label_data=unique(data1,return_inverse='True')
cnf_matrix1 = confusion_matrix(label_data, final)
fpr1, tpr1, thresholds1 = metrics.roc_curve(label_data,final, pos_label=1)
acc1=float(sum(label_data==final))/len(final)
stop=timeit.default_timer()
time1=stop-start

start1=timeit.default_timer()
db = DBSCAN(eps=0.005, min_samples=10).fit(sim_matrix)
labels1 = db.labels_
n_clusters_ = len(set(labels1)) - (1 if -1 in labels1 else 0)
final1=[1 if x==-1 else x for x in labels1]
stop1=timeit.default_timer()
time2=stop1-start1

cnf_matrix2 = confusion_matrix(label_data, final1)
fpr2, tpr2, thresholds1 = metrics.roc_curve(label_data,final1, pos_label=1)
acc2=float(sum(label_data==final1))/len(final1)

start2=timeit.default_timer()


model = AgglomerativeClustering(n_clusters=2,linkage="average", affinity='euclidean',connectivity='array-like')
model.fit(sim_matrix)
final3=model.labels_
cnf_matrix3 = confusion_matrix(label_data, final3)
fpr3, tpr3, thresholds1 = metrics.roc_curve(label_data,final3, pos_label=1)
acc3=float(sum(label_data==final3))/len(final3)
stop2=timeit.default_timer()
time3=stop2-start2

start3=timeit.default_timer()
birch = Birch(n_clusters=2,branching_factor=60,threshold=0.000005)
birch.fit(sim_matrix)
final4=birch.predict(sim_matrix)
cnf_matrix4 = confusion_matrix(label_data, final4)
fpr4, tpr4, thresholds1 = metrics.roc_curve(label_data,final4, pos_label=1)
acc4=float(sum(label_data==final4))/len(final4)
stop3=timeit.default_timer()
time4=stop3-start3

start4=timeit.default_timer()
spclust= SpectralClustering(n_clusters=2, eigen_solver='lobpcg', n_init=10, affinity='rbf')
spclust.fit(sim_matrix)
final5=spclust.labels_
acc5=float(sum(label_data==final5))/len(final5)
stop4=timeit.default_timer()
time5=stop4-start4
print time1 
print time2 
print time3

#cent3=raw_input('initial point for cluster')
#cent3=np.int(cent3)
#score3=[]
#for i in range(0,len(data2)):
# score3.append(similar(data2[i],data2[cent3]))

#def convert_to_ascii(text):
# return " ".join(str(ord(char)) for char in text)


#asci_text=[]
#for i in range(0,len(data2)):
# asci_text.append(convert_to_ascii(data2[i]))

#asci_text1=np.array(asci_text)


#token_word=[]
#for i in range(0,len(data2)):
# c=data2[i].split(' ')
# c=np.array(c)
# token_word=np.array(token_word)
# token_Word=np.array(token_word)
# token_word=np.append(token_word,c)
#import re
#len(token_word)
#token_word1=np.str(token_word)
#token_word1
#token_word[0]
#token_word[-1]
#token_word[-2]

#token_word
#c[-1]
#token_word[-5]
#token_word[-4]
#token_word[-6]
#len(token_word1)
#data2
#data2[0]
#token_word1=np.str(data2)
#len(token_word)
#len(token_word1)
#new_string
#import readline
#readline.write_history_file('docum1.py')


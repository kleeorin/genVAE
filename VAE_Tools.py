#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 00:25:00 2020

@author: yaakov
"""

import numpy as np
from Bio import SeqIO
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.io import loadmat

def one_hot(a,q):
    # get a numerical np vector of integers 0:(q-1) and return its one-hot representation 
    b = np.zeros((a.size, q))
    b[np.arange(a.size),a] = 1
    return b.flatten()



def reverse_one_hot(a,q):
    # get a one-hot np vector of integers 0,1 and return its numerical representation 
    b = np.zeros((int(a.size/q)))
    b[(a.nonzero()[0]//q)]=(a.nonzero()[0]%q);
    return b



def hardmax(pred_oh,q):
    #input a vector on one-hot format with various values
    #return a vector on one-hot format with a modulo q hardmax
    return one_hot(np.argmax(np.reshape(pred_oh,(-1,q)),axis=1),q)



def load_fasta(file):
    # input fasta file containing the MSA
    # return MSA_ohr - one hot representation of the MSA
    # and MSA the regualr representation of the MSA
    
    # Amino acid code and error letters
    code = "-ACDEFGHIKLMNPQRSTVWY"
    q=len(code);
    AA_to_num=dict([(code[i],i) for i in range(len(code))])
    errs = "BJOUXZabcdefghijklmonpqrstuvwxyz"
    AA_to_num.update(dict([(errs[i],-1) for i in range(len(errs))]))
    
    # Parse sequence contacts of fasta into MSA
    MSA=np.array([])
    for record in SeqIO.parse(file, "fasta"):
        seq=np.array([[AA_to_num[record.seq[i]] for i in range(len(record.seq))]])
        if MSA.shape[0]==0:
            MSA=seq;
        else:
            MSA=np.append(MSA,seq,axis=0)
            
    # Remove all errornous sequences (contain '-1')
    MSA=np.delete(MSA,(np.sum(MSA==-1,axis=1)).nonzero()[0][0],axis=0)
    
    # Transfer to One-Hot representation
    MSA_oh=np.array([one_hot(a,q) for a in MSA])
    
    # Create training and cross-valid datasets
    order=np.random.permutation(MSA_oh.shape[0])
    MSA_oh=MSA_oh[order,:]
    MSA=MSA[order,:]
    return MSA_oh, MSA



def read_bm(file):
    # input filename from BM-DCA algorithm in a specific format
    # return non-one-hot representation of h and J full matrices.
    tab=pd.read_csv(file, delimiter=' ',names=['var','i','j','a','b','val'])
    Js=tab.loc[tab['var']=='J']
    Js=Js.astype({'a':int,'b':int});
    hs=tab.loc[tab['var']=='h']
    hs=hs.drop(columns=['b','val'])
    hs=hs.rename(columns={"j":"a","a":"val"})                          
    
    #initialize the output variables based on the size of the model
    L=max(Js.j.max(),Js.i.max())+1;
    q=Js.a.max()+1;
    J=np.zeros((L,L,q,q));
    h=np.zeros((L,q));
    NJ=Js.shape[0]
    Nh=hs.shape[0]
    
    k=[i for i in range(NJ)]
    J[Js.i[k],Js.j[k],Js.a[k],Js.b[k]]=Js.val[k]
    # this is both halves of J!
    J[Js.j[k],Js.i[k],Js.b[k],Js.a[k]]=Js.val[k]
    k=[i+NJ for i in range(Nh)]
    h[hs.i[k],hs.a[k]]=hs.val[k]
    
    return h,J

def read_mat_model(file):
    # input filename from BM-DCA algorithm in a specific format
    # return non-one-hot representation of h and J full matrices.
    x=loadmat(file)
    J=x['J'];
    h=x['h']
    J=J.transpose(2,3,0,1)
    h=h.transpose(1,0)
    return h,J
    

def DCA_energy(seqs,h,J):
    # given an MSA (must be matrix) and h,J model parameters
    # output the energy of per sequence in the MSA
    L=seqs.shape[1]
    N=seqs.shape[0]
    energy=np.zeros(N);
    for k in range(N):
        energy[k]=energy[k]+np.sum(np.array([h[i,seqs[k,i]] for i in range(L)]))
        energy[k]=energy[k]+np.sum(np.sum(np.array([[J[i,j,seqs[k,i],seqs[k,j]] for j in range(i+1,L)] for i in range(L)])))

    return energy


def diveregnce_from(MSA,MSA_0):
    # input MSA and a reference MSA_0
    # output a vector of distances (0-1) from closest reference seq
    div_vector=np.zeros(MSA.shape[0])
    for i in range(MSA.shape[0]):
        div_vector[i]=1-np.max(np.sum(MSA[i,:]==MSA_0,axis=-1))/MSA.shape[1]
    div=np.average(div_vector)
    return div, div_vector

def diveregnce_from_self(MSA):
    # input MSA and a reference MSA_0
    # output a vector of distances (0-1) from closest reference seq
    div_vector=np.zeros(MSA.shape[0])
    for i in range(MSA.shape[0]):
        div_vector[i]=1-np.max(np.concatenate((np.sum(MSA[i,:]==MSA[0:i,:],axis=-1),np.sum(MSA[i,:]==MSA[i+1:,:],axis=-1))))/MSA.shape[1]
    div=np.average(div_vector)
    return div, div_vector
def variability(MSA):
    return np.mean(pdist(MSA, metric='hamming'))
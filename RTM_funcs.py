from realSim import simulator_base,downsample
from realSim_fake import simulator_base_fake
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm.notebook import trange, tqdm
from joblib import Parallel, delayed
import warnings
import scipy.stats as stats
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

np.seterr(divide='ignore', invalid='ignore')
np.seterr(invalid='ignore')

def make_one_trace(t,x,c,lab):
    plt.plot(t,x,color=c,alpha=0.8,label=lab)
    xmean = np.mean(x)
    xs = np.std(x)
    xlo = xmean - 1.96*xs
    xhi = xmean + 1.96*xs
    plt.plot([t[0],t[-1]],[xlo,xlo],'--',color=c,alpha=0.5)
    plt.plot([t[0],t[-1]],[xmean,xmean],':',color=c,alpha=0.5)
    plt.plot([t[0],t[-1]],[xhi,xhi],'--',color=c,alpha=0.5)
    
def drawDefinitions(): 
    yrs = 10
    samps = 12*yrs
    t = np.linspace(0,yrs,samps)
    x1 = np.random.randn(samps)*.5 + 2
    x2 = np.random.randn(samps)*.5 + 3.5
    x3 = np.random.randn(samps)*.8 + 5
    x4 = np.random.randn(samps)*.5 + 8

    plt.figure(figsize=(10,6))

    make_one_trace(t,x4,'green','A: Always qualified')
    make_one_trace(t,x3,'magenta','B: Usually qualified')
    make_one_trace(t,x2,'orange','C: Sometimes qualified')
    make_one_trace(t,x1,'red','D: Never qualified')

    plt.plot([t[0],t[-1]],[4,4],'--k',linewidth=6,label='threshold')
    plt.xlabel('Years')
    plt.ylabel('Seizures per month')
    plt.legend(loc='upper right')
    plt.show()

def organize_all_sets():
    fn_pre = 'RTM'
    bigP = pd.DataFrame()
    for hist in [1,2,3,6,12]:
        for baseTF in [True,False]:
            fn = f'{fn_pre}_h{hist}_b{baseTF}.csv'
            p = pd.read_csv(fn)
            p['hist'] = [hist]*len(p)
            p['baseTF'] = [baseTF]*len(p)
            bigP = pd.concat([bigP,p])
    print(bigP)         
    bigP.to_csv('RTM-allh.csv',index=False)
 
def doAllsetsHandB(N):
    for hist in [1,2,3,6,12]:
        for baseTF in [True,False]:
            trySomeCr(N=N,hist=hist,baseTF=baseTF)

    
def trySomeCr(N,hist,baseTF):
    fn_pre = 'RTM'
    CrList = [0,1,2,3,4,5,6,7,8]
    dd = pd.DataFrame({})
    fn = f'{fn_pre}_h{hist}_b{baseTF}.csv'
    for Cr in tqdm(CrList):
        d = howManyRTM2(Cr=Cr,N=N,hist=hist,baseTF=baseTF)
        dd = pd.concat([dd,d],ignore_index=True)
    print(f'Filename: {fn} N={N} hist={hist} baseTF={baseTF}')
    print(dd) 
    dd.to_csv(fn,index=False)

           
def howManyRTM2(Cr,N,hist,baseTF):
    ## Constants here
    sampRATE=1
    #Cr=4
    yrs = 2
    number_of_days=30*12*yrs + (hist*30)
    #N = 10000
    
    numCPUs = 9
    with Parallel(n_jobs=numCPUs, verbose=False) as par:
        temp = par(delayed(inner_loop)(sampRATE,number_of_days,Cr,hist,baseTF) for _ in range(N))
    temp2 = np.array(temp,dtype=float)
    RTMtypes = np.sum(temp2[:,0:4],axis=0)
    # I expect to see RuntimeWarnings in this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        PC_a = np.nanmedian(temp2[temp2[:,0]>0,4])
        PC_b = np.nanmedian(temp2[temp2[:,1]>0,4])
        PC_c = np.nanmedian(temp2[temp2[:,2]>0,4])
        PC_d = np.nanmedian(temp2[temp2[:,3]>0,4])
        RR50_a = 100*np.nanmean(temp2[temp2[:,0]>0,5])
        RR50_b = 100*np.nanmean(temp2[temp2[:,1]>0,5])
        RR50_c = 100*np.nanmean(temp2[temp2[:,2]>0,5])
        RR50_d = 100*np.nanmean(temp2[temp2[:,3]>0,5])
        MPC = np.nanmedian(temp2[:,4])
        RR50 = 100*np.nanmean(temp2[:,5])
            
    df = pd.DataFrame({'Cr':[Cr],
                       'Type A':[int(np.round(100*RTMtypes[0]/N))],
                        'Type B':[int(np.round(100*RTMtypes[1]/N))],
                        'Type C':[int(np.round(100*RTMtypes[2]/N))],
                        'Type D':[int(np.round(100*RTMtypes[3]/N))],
                        'MPC A':[PC_a],
                        'MPC B':[PC_b],
                        'MPC C':[PC_c],
                        'MPC D':[PC_d],
                        'RR50 A':[RR50_a],
                        'RR50 B':[RR50_b],
                        'RR50 C':[RR50_c],
                        'RR50 D':[RR50_d],
                        'MPC X':[MPC],
                        'RR50 X':[RR50]})
 
    return df

def inner_loop(sampRATE,number_of_days,Cr,hist,baseTF):
    baseline_dur = 2  # this is an assumed constant for now
    test_dur = 3       # this is an assumed constant for now
    rct_dur = baseline_dur + test_dur
    
    RTMtypes = np.zeros(6)
    seizure_diary_final,mSF,overdispersion,rate,modulated_rate,theFreqs,theAmps,cycles,do_clusters,modulated_cluster_rate,seizure_diary_A,seizure_diary_B = simulator_base(sampRATE,number_of_days, returnDetails = True)
    monthly_diary = downsample(seizure_diary_final,30)
    mSF = mSF[0]
    #print(monthly_diary)
    true_eligible = mSF>= Cr
    loEnd = np.mean(monthly_diary) - 1.96*np.std(monthly_diary)
    hiEnd = np.mean(monthly_diary) + 1.96*np.std(monthly_diary)

    if true_eligible:
        if (loEnd >= Cr):
            # type A
            RTMtypes[0]+=1
        else:
            # type B
            RTMtypes[1]+=1
    else:
        if (hiEnd >= Cr):
            # type C
            RTMtypes[2]+=1
        else:
            # type D            
            RTMtypes[3]+=1
        
    
    RTMtypes[4:6] = [np.NaN,np.NaN]
    if (hist>0) and (baseTF==False):
        # only history here
        pmd = pd.DataFrame({'x':monthly_diary})
        temp = pmd.rolling(hist).mean()
        running_ave = np.array(temp.x)
        w01 = np.where(running_ave>Cr)
        w0 = w01[0]
        if len(w0)>0:
            if w0[0]<(len(monthly_diary)-rct_dur):
                ind = w0[0]
                baseline_sz = monthly_diary[ind:ind+baseline_dur]
                test_sz = monthly_diary[ind+baseline_dur:ind+rct_dur]
                B = np.mean(baseline_sz)
                T = np.mean(test_sz)
                PC = 100*np.divide(B-T,B, out=np.zeros(B.shape, dtype=float), where=B!=0)
                rr50 = PC>=50
                RTMtypes[4:6] = [PC,rr50]
                
    elif (baseTF==True):
        # use baseline and hist
        pmd = pd.DataFrame({'x':monthly_diary})
        temp = pmd.rolling(hist+baseline_dur).mean()
        running_ave = np.array(temp.x)
        w01 = np.where(running_ave>Cr)
        w0 = w01[0]
        if len(w0)>0:
            if w0[0]<(len(monthly_diary)-rct_dur + baseline_dur - hist):
                ind = w0[0]-baseline_dur+1
                baseline_sz = monthly_diary[ind:(ind+baseline_dur)]
                test_sz = monthly_diary[(ind+baseline_dur):(ind+rct_dur)]
                B = np.mean(baseline_sz)
                T = np.mean(test_sz)
                PC = 100*np.divide(B-T,B, out=np.zeros(B.shape, dtype=float), where=B!=0)
                rr50 = PC>=50
                RTMtypes[4:6] = [PC,rr50]  
  
   
    return RTMtypes





def draw_bigger_summary():
    # assumes you already calculated doAllsetsHandB()
    #rr50list = np.zeros((5,2,9))
    #mpclist = np.zeros((5,2,9))
    fn_pre = 'RTM'
    Crlist = np.array([0,1,2,3,4,5,6,7,8])
    someOnes = np.ones(9)
    d = pd.DataFrame({})
    for hi,hist in enumerate([1,2,3,6,12]):
        for bi,baseTF in enumerate([True,False]):
            fn= f'{fn_pre}_h{hist}_b{baseTF}.csv'
            txt= f'Baseline={baseTF} history={hist}'
            x = pd.read_csv(fn)
            #mpclist[hi,bi,:] = x['MPC X']
            #rr50list[hi,bi,:] = x['RR50 X']
            d2 = pd.DataFrame({'history':hist*someOnes,'baseTF':baseTF*someOnes,
                               'Cr':Crlist,'MPC':x['MPC X'],'RR50':x['RR50 X']})
            d = pd.concat([d,d2],ignore_index=True)
    
    plt.figure(figsize=(10,5))
    plt.subplots(nrows=2,ncols=1,sharex=True)
    plt.subplot(2,1,1)   
    sns.scatterplot(data=d,x='Cr',y='RR50',style='baseTF',
            palette=['black','purple','red','blue','green'],
            hue='history',alpha=0.8,legend=False,s=50,
            markers=['x','+'])
    pList=['black','purple','red','blue','green']
    lineList=['-',':']
    for hi,hist in enumerate([1,2,3,6,12]):
        for bi,baseTF in enumerate([True,False]):
            x2 = d[d['history']==hist]
            x3 = x2[x2['baseTF']==baseTF]
            lineTag = lineList[bi]
            plt.plot(x3['Cr'],x3['RR50'],lineTag,color=pList[hi],alpha=0.8)
                 
    #plt.legend(loc='lower right')
    #plt.xlabel('Threshold seizures/month')
    plt.ylabel('RR50 %')
    plt.title('RR50')
    plt.subplot(2,1,2)   
    sns.scatterplot(data=d,x='Cr',y='MPC',style='baseTF',
            palette=['black','purple','red','blue','green'],
            hue='history',alpha=0.8,s=50,
            markers=['x','+'])
    for hi,hist in enumerate([1,2,3,6,12]):
        for bi,baseTF in enumerate([True,False]):
            x2 = d[d['history']==hist]
            x3 = x2[x2['baseTF']==baseTF]
            lineTag = lineList[bi]
            plt.plot(x3['Cr'],x3['MPC'],lineTag,color=pList[hi],alpha=0.8)
       
    plt.legend(bbox_to_anchor=(1.02, 1.2), loc='upper left', borderaxespad=0)
    plt.xlabel('Threshold seizures/month')
    plt.ylabel('MPC %')
    plt.title('MPC')
    
    plt.show()
            
def drawAllsetsHandB():
    fn_pre = 'RTM'
    for hist in [1,2,3,6,12]:
        for baseTF in [True,False]:
            fn= f'{fn_pre}_h{hist}_b{baseTF}.csv'
            txt= f'Baseline={baseTF} history={hist}'
            draw_RTM_summary(fn=fn,title_pre=txt)

def drawSomesetsHandB():
    fn_pre = 'RTM'


    plt.figure(figsize=(8,15))
    plt.subplots(nrows=3,ncols=2,sharey=True,sharex=True)
    for hi,hist in enumerate([1,3,12]):
        for baseTF in [False]:
            fn= f'{fn_pre}_h{hist}_b{baseTF}.csv'
            title_pre= f'History={hist}'
            #draw_RTM_summary(fn=fn,title_pre=txt,do_show=False)
            x = pd.read_csv(fn)
            plt.subplot(3,2,hi*2+1)
            drawOneLine(x,'MPC','A','green')
            drawOneLine(x,'MPC','B','magenta')
            drawOneLine(x,'MPC','C','orange')
            drawOneLine(x,'MPC','X','black')
            plt.grid(True)
            plt.ylim([-5,80])
            if hi==2:
                plt.xlabel('Threshold seizures/month')
                plt.ylabel('MPC %')
            plt.title(f'{title_pre}')
            plt.subplot(3,2,hi*2+2)
            drawOneLine(x,'RR50','A','green')
            drawOneLine(x,'RR50','B','magenta')
            drawOneLine(x,'RR50','C','orange')
            drawOneLine(x,'RR50','X','black')
            plt.grid(True)
            #drawOneLine(x,'RR50','D','red')
            plt.ylim([-5,80])
            if hi==2:
                plt.xlabel('Threshold seizures/month')
                plt.ylabel('RR50 %')
                plt.legend(bbox_to_anchor=(1.02, 1.2), loc='upper left', borderaxespad=0)

            plt.title(f'{title_pre}')
    plt.show()
             
def draw_RTM_summary(fn,title_pre,do_show=True):
    x = pd.read_csv(fn)
    plt.figure(figsize=(10,3))
    plt.subplot(1,2,1)
    drawOneLine(x,'MPC','A','green')
    drawOneLine(x,'MPC','B','magenta')
    drawOneLine(x,'MPC','C','orange')
    drawOneLine(x,'MPC','X','black')
    #drawOneLine(x,'MPC','D','red')
    plt.grid(True)
    plt.ylim([-5,80])
    plt.xlabel('Threshold seizures/month')
    plt.ylabel('MPC %')
    plt.title(f'{title_pre} MPC')
    plt.subplot(1,2,2)
    drawOneLine(x,'RR50','A','green')
    drawOneLine(x,'RR50','B','magenta')
    drawOneLine(x,'RR50','C','orange')
    drawOneLine(x,'RR50','X','black')
    plt.grid(True)
    #drawOneLine(x,'RR50','D','red')
    plt.ylim([-5,80])
    plt.xlabel('Threshold seizures/month')
    plt.ylabel('MPC %')
    plt.title(f'{title_pre} RR50')
    plt.legend()
    if do_show==True:
        plt.show()
    
def drawOneLine(x,metrictype,letter,c):
    lab = f'{metrictype} {letter}'
    s_lab = f'Type {letter}'
    
    if letter=='X':
        useInds = x['Type A']>-1
    else:
        useInds = x[s_lab]>0
    xset = x['Cr']
    yset = x[lab]
    if np.sum(useInds)>0:
        plt.plot(xset[useInds],yset[useInds],'-',label=lab,color=c)
    L = x.shape[0]
    y = x[lab]
    if letter != 'X':
        slist = x[s_lab]
        
    sums = x['Type A'] + x['Type B'] + x['Type C']
    for i in range(L):
        if letter != 'X':
            Msize = 10 * slist.iloc[i]/sums.iloc[i]
        else:
            Msize = 2
        plt.plot(x.Cr.iloc[i],y.iloc[i],'o',markersize=Msize,color=c,alpha=0.8)
        
        
def test_this_idea(howMany=10000,DRG=0.3,N=200,baseTF=False,doingFAKE=False):
    # make an RCT sim to prove this is better
    tflist = [False,True]
    #for N in [200,250]:
    for i,thisHist in enumerate([1,2,3,6,12]):
        #baseTF = False       
        runSet_with_HandB(howMany=howMany,hist=thisHist,baseTF=baseTF,minSz=4,N=N,years=3,DRG=DRG,doPIX=False,doingFAKE=doingFAKE)

def runSet_with_HandB(howMany,hist,baseTF,minSz,N,years,DRG=.2,PCB=0,baseline=2,test=3,numCPUs=9,doPIX=True,printTF=True,doingFAKE=False,saveFile=False,saveFn=''):
    #fname='Fig5-RCT.tiff'
    with Parallel(n_jobs=numCPUs, verbose=False) as par:
        X = par(delayed(run1trial_with_hist_and_base)(minSz,N,DRG,PCB,baseline,test,hist,baseTF,years,doingFAKE) for _ in trange(howMany,leave=False))
    X2 = np.array(X,dtype=float)
    X3 = X2[:,0:4].copy()
    #rr50_pow = np.round(100*np.mean(X2[:,4]))
    #mpc_pow = np.round(100*np.mean(X2[:,5]))
    rr50_pow = np.mean(X2[:,4])
    mpc_pow = np.mean(X2[:,5])
    if saveFile==True:
        np.savetxt(saveFn, X2[:,4:6], delimiter=',', newline='\n', header='rr50_yes,mpc_yes')
    if doPIX==True:
        plt.boxplot(X3)
        plt.xticks([1,2,3,4],['RR50_placebo','RR50_drug','MPC_placebo','MPC_drug'])
        #plt.savefig(fname,dpi=600)
        plt.show()
    if printTF==True:
        print(f'N={N} hist={hist} baseTF={baseTF} minSz={minSz} drg={DRG} -> RR50 = {rr50_pow} MPC = {mpc_pow}')
        #for silly in range(4):
        #    y = X3[:,silly]
        #    print(f'X[{silly}]: mean = {np.mean(y):4.4} median = {np.median(y):4.4} std dev={np.std(y):4.4}')
        print(f'({(np.mean(X3[:,1])-np.mean(X3[:,0])):4.4}: {np.mean(X3[:,0]):4.4}+-{np.std(X3[:,0]):4.4} {np.mean(X3[:,1]):4.4}+-{np.std(X3[:,1]):4.4}')
    #ss1 = np.std(X3[:,1])/100
    #ss0 = np.std(X3[:,0])/100
    #Rp = 100*fast_calc_rr50(N,10000,ss1,np.mean(X3[:,1])/100,ss0,np.mean(X3[:,0])/100,numCPUs)
    #print(f'Rp = {Rp} diff = {np.abs(rr50_pow-Rp):.2}')
    #ss1 = np.mean([ss1,ss0])
    #ss0 = ss1
    #ss1 = np.std(X3[:,1])/100
    #ss0 = np.std(X3[:,0])/100
    #Rp = 100*fast_calc_rr50(N,10000,ss1,np.mean(X3[:,1])/100,ss0,np.mean(X3[:,0])/100,numCPUs)
    #print(f'Rp = {Rp} diff = {np.abs(rr50_pow-Rp):.2}')
    #return X2
    diffR = np.mean(X2[:,1])-np.mean(X2[:,0])
    diffM = np.mean(X2[:,3])-np.mean(X2[:,2])
    PCB_M = np.mean(X2[:,2])
    PCB_R = np.mean(X2[:,0])
    return rr50_pow,mpc_pow,diffR,diffM,PCB_R,PCB_M

def run1trial_with_hist_and_base(minSz,N,DRG,PCB,baseline,test,hist,baseTF,years,doingFAKE):
    trialData = makeTrial_with_hist_and_base(minSz,N,DRG,PCB,baseline,test,hist,baseTF,years,doingFAKE)
    PC = getPC(trialData,baseline,test)
    nover2 = int(N/2)
    PC_pcb = PC[:nover2]
    PC_drg = PC[nover2:]
    rr50_pcb = RR50(PC_pcb)
    rr50_drg = RR50(PC_drg)
    mpc_pcb = MPC(PC_pcb)
    mpc_drg = MPC(PC_drg)
    p_rr50 = calculate_fisher_exact_p_value(PC_pcb,PC_drg)<0.05
    p_mpc = calculate_MPC_p_value(PC_pcb,PC_drg)<0.05
    return [rr50_pcb,rr50_drg,mpc_pcb,mpc_drg,p_rr50,p_mpc]

def RR50(PC):
    return 100*np.mean(PC>=50)

def MPC(PC):
    return np.median(PC)

def getPC(trialData,baseline,test):
    # assumes by month
    B = np.sum(trialData[:,:baseline],axis=1) / (baseline)
    T = np.sum(trialData[:,baseline:],axis=1) / (test)
    PC = 100*np.divide(B-T,B, out=np.zeros(B.shape, dtype=float), where=B!=0)
    #PC = 100*(B-T)/B
    return PC

def makeTrial_with_hist_and_base(minSz,N,DRG,PCB,baseline,test,hist,baseTF,years,doingFAKE):
    dur = baseline+test

    trialData = np.zeros((N,dur))
    for pt in range(N):
        temp1 = makeOnePt_with_hist_and_base(minSz,hist,baseTF,dur,baseline,years,doingFAKE)
        #temp1[1] = temp1[0]
        #temp1[2:] = temp1[3]*np.ones(3)
        # apply placebo to both groups
        temp2 = applyDrug(PCB,temp1,baseline)
        if pt>=(N/2):
            # apply drug here
            trialData[pt,:] = applyDrug(DRG,temp2,baseline)
        else:
            # this is placebo group, so do nothing else
            trialData[pt,:] = temp2.copy()
    
    return trialData

def makeOnePt_with_hist_and_base(minSz,hist,baseTF,dur,baseline,years,doingFAKE=False):
    
    sampFREQ = 1
    extra_dur = years*12
    full_months = extra_dur + dur
    full_dur = int(full_months*30)
    if baseTF==True:
        #max_window = hist_dur + baseline
        max_window = hist + baseline
        max_ind = full_months - (dur-baseline)
        sub1 = baseline - 1
    else:
        #max_window = hist_dur
        max_window = hist
        max_ind = full_months - dur
        sub1 = 0
    
    minSz_tot = minSz*hist
    
    notDone = True
    while (notDone==True):
        if doingFAKE:
            x=simulator_base_fake(sampRATE=sampFREQ,number_of_days=full_dur)
        else:
            x = simulator_base(sampRATE=sampFREQ,number_of_days=full_dur)
        x2 = downsample(x,int(sampFREQ*30))
        # make a moving window, try to qualify
        p = pd.DataFrame({'x2':x2}).rolling(max_window).sum()
        p2 = np.array(p)
        # where was a qualifier?
        w01 = np.where(p2>minSz_tot)
        w1 = w01[0]
        if len(w1)>0:
            if w1[0] < max_ind:
                # keep only the months of the trial itself
                x3 = x2[(w1[0]-sub1):(w1[0]+dur-sub1)].copy()
                notDone = False

    return x3

def applyDrug(efficacy,x,baseline):
    # INPUTS:
    #  efficacy = fraction of seziures removed
    #  x = diary
    #  baseline = number of samples to consider as baseline samples
    #     that do not get drug applied at all
    
    # put some jitter into the efficacy
    #thisEfficacy = np.maximum(efficacy + np.random.randn()*0.05,0)
    if 1:
        # USE THIS DEFAULT STRATEGY
        allS = np.sum(x[baseline:])
        #deleter = np.random.random(int(allS))<thisEfficacy
        deleter = np.random.random(int(allS))<efficacy

        x2 = x.copy()
        counter=0
        for iter in range(baseline,len(x)):
            for sCount in range(int(x[iter])):
                x2[iter] -= deleter[counter]
                counter += 1
    else:
        # simplest model for applyDrug has no nuance...
        x2 = x.copy()
        x2[baseline:] = x[baseline:]*(1-efficacy)    
    return x2



def calculate_fisher_exact_p_value(placebo_arm_percent_changes,
                                   drug_arm_percent_changes):

    num_placebo_arm_responders     = np.sum(placebo_arm_percent_changes >= 50)
    num_drug_arm_responders        = np.sum(drug_arm_percent_changes    >= 50)
    num_placebo_arm_non_responders = len(placebo_arm_percent_changes) - num_placebo_arm_responders
    num_drug_arm_non_responders    = len(drug_arm_percent_changes)    - num_drug_arm_responders

    table = np.array([[num_placebo_arm_responders, num_placebo_arm_non_responders], [num_drug_arm_responders, num_drug_arm_non_responders]])

    [_, RR50_p_value] = stats.fisher_exact(table)

    return RR50_p_value

def calculate_MPC_p_value(placebo_arm_percent_changes,
                                     drug_arm_percent_changes):

    # Mann_Whitney_U test
    [_, MPC_p_value] = stats.ranksums(placebo_arm_percent_changes, drug_arm_percent_changes)

    return MPC_p_value


def do_a_RR50_test(numCPUs=9):
    reps = 10000
    N = 200
    drugDiff = .2
    
    slist = [0.03,0.04,0.05,0.06]
    rlist = np.zeros((5,30))
    rlist[0,:] = np.arange(.05,.34,.01)
    for Pi,PCBm in tqdm(enumerate(rlist[0,:])):
        DRGm = PCBm+.2
        for si,s in enumerate(slist):
            rlist[si+1,Pi] = fast_calc_rr50(N,reps,s,DRGm,s,PCBm,numCPUs)
    for si,s in enumerate(slist):    
        plt.plot(rlist[0,:],rlist[si+1,:],'x-',label=f'StdDev={s}')
    
    plt.xlabel('Mean placebo strength')
    plt.ylabel('RCT power')
    plt.legend()
    plt.title('Assuming 20% higher drug')
    plt.show()
    difflist = [.15,.20,.25,.30]
    rlist = np.zeros((5,30))
    rlist[0,:] = np.arange(.05,.34,.01)
    s = 0.05
    for Pi,PCBm in tqdm(enumerate(rlist[0,:])):
        for di,d in enumerate(difflist):
            DRGm = PCBm+d
            rlist[di+1,Pi] = fast_calc_rr50(N,reps,s,DRGm,s,PCBm,numCPUs)
    for di,d in enumerate(difflist):    
        plt.plot(rlist[0,:],rlist[di+1,:],'x-',label=f'DrugDifference={d}')
    
    plt.xlabel('Mean placebo strength')
    plt.ylabel('RCT power')
    plt.legend()
    plt.title('Assuming stddev=0.05')
    plt.show()
        
def fast_calc_rr50(N,reps,DRGs,DRGm,PCBs,PCBm,numCPUs):
    Narm = int(N/2)
    thisDrugResp = Narm*np.minimum(np.maximum((np.random.randn(reps)*DRGs + DRGm),0),1)
    thisDrugResp = np.maximum(thisDrugResp,0)
    thisDrugResp = np.minimum(thisDrugResp,Narm)
    thisDrugNonResp = Narm - thisDrugResp
    thisPCBresp = Narm*np.minimum(np.maximum((np.random.randn(reps)*PCBs + PCBm),0),1)
    thisPCBresp = np.maximum(thisPCBresp,0)
    thisPCBresp = np.minimum(thisPCBresp,Narm)
    thisPCBnonresp = Narm - thisPCBresp
    with Parallel(n_jobs=numCPUs, verbose=False) as par:
        temp = par(delayed(doRR50_loop)(thisPCBresp,thisPCBnonresp,thisDrugResp,thisDrugNonResp,thisTrial) for thisTrial in range(reps))
    rr50sum = np.mean(np.array(temp,dtype=int))
    return rr50sum
        
        
def doRR50_loop(thisPCBresp,thisPCBnonresp,thisDrugResp,thisDrugNonResp,thisTrial):
    num_placebo_arm_responders = thisPCBresp[thisTrial]
    num_placebo_arm_non_responders = thisPCBnonresp[thisTrial]
    num_drug_arm_responders =  thisDrugResp[thisTrial]
    num_drug_arm_non_responders = thisDrugNonResp[thisTrial]
    table = np.array([[num_placebo_arm_responders, num_placebo_arm_non_responders], [num_drug_arm_responders, num_drug_arm_non_responders]])
    [_, RR50_p_value] = stats.fisher_exact(table)
    rr50 = (RR50_p_value<0.05)
    return rr50

def randrange(n,vmin,vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin

def build_the_map(numCPUs=9):
    samps = 1000
    N=200

    x = randrange(samps,0.05,.5) # placebo mean
    y = randrange(samps,0.1,.4) # drug difference
    f = randrange(samps,0.02,.07) # std ave
    z = np.zeros(samps)
    #for i in trange(samps):
        #print(f'x={x[i]} y={y[i]} f={f[i]}')
        #z[i] = fast_calc_rr50(N=200,reps=10000,DRGs=f[i],DRGm=x[i]+y[i],PCBs=f[i],PCBm=x[i],numCPUs=numCPUs)
    z = [ fast_calc_rr50(N=200,reps=10000,DRGs=f[i],DRGm=x[i]+y[i],PCBs=f[i],PCBm=x[i],numCPUs=numCPUs) for i in trange(samps)]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, c=f)
    plt.show()
    df = pd.DataFrame({'x':x,'y':y,'z':z,'f':f})
    df.to_csv('themap.csv',index=False)
    
def make_a_model_of_pow():
    df = pd.read_csv('themap.csv')

    regr = linear_model.LinearRegression()
    regr.fit(df[['y','x','f']], df['z'])


# figure out what happens to power as eligibility changes
def test_out_power(fname,reps=1000,numCPUs=9,baseTF=False,doingFAKE=False):
    #DRG=0.3
    
    counter=0
    theMode = 'w'
    for N in [100,200,300,600]:
        for DRG in [0.2,0.3,0.4]:
            for thisHist in [1,3,6,12]:
                for minSz in [1,2,3,4,5,6,7,8]:
                    rr50_pow,mpc_pow,diffR,diffM,PCB_R,PCB_M = runSet_with_HandB(howMany=reps,hist=thisHist,baseTF=baseTF,minSz=minSz,N=N,years=3,DRG=DRG,numCPUs=numCPUs,doPIX=False,printTF=False,doingFAKE=doingFAKE)
                    d1 = pd.DataFrame({'N':[N],'DRG':[DRG],'history':[thisHist],'minSz':[minSz],
                        'rr50_pow':[rr50_pow],'mpc_pow':[mpc_pow],
                        'diffR':[diffR],'diffM':[diffM],'PCB_RR50':[PCB_R],'PCB_MPC':[PCB_M]})
                    
                    d1.to_csv(fname,header=(counter==0),mode=theMode,index=False)
                    print(d1)
                    counter+=1
                    theMode = 'a'


def show_different_graphs(fn):
    
    x = pd.read_csv(fn)
    x.rename(columns = {'hist':'history'}, inplace = True)
    #print(x)
    Dlist = [0.2,0.3,.4]



    #pairsRR = np.array([[600,.2],[200,.3],[100,.4]])
    #pairsMP = np.array([[300,.2],[100,.3],[100,.4]])
    pairsRR = np.array([[300,.2],[100,.3],[100,.4]])
    pairsMP = np.array([[100,.2],[100,.3],[100,.4]])
    for PCBflag in [0,1,2,3]:
        plt.subplots(nrows=2,ncols=3,sharex=True,sharey=True)
        for p in range(3):
            N = pairsRR[p,0]
            DRG = pairsRR[p,1]
            x1 = x[np.bitwise_and(x['N']==N,x['DRG']==DRG)]
            
            
            plt.subplot(2,3,1+p)
            if PCBflag==1:
                sns.scatterplot(data=x1,x='PCB_RR50',y='rr50_pow',hue='minSz',size='minSz',style='history',legend=False,palette='viridis',alpha=0.6)
                plt.xlim([0,40])
                plt.ylim([.7,1])
            elif PCBflag==0:
                sns.scatterplot(data=x1,x='minSz',y='PCB_RR50',hue='minSz',size='minSz',style='history',legend=False,palette='viridis',alpha=0.6)
                plt.xlim([0, 10])
                plt.ylim([0,40])
            elif PCBflag==2:
                sns.scatterplot(data=x1,x='diffR',y='rr50_pow',hue='minSz',size='minSz',style='history',legend=False,palette='viridis',alpha=0.6)
                plt.xlim([0,40])
                plt.ylim([.7,1])
            elif PCBflag==3:
                sns.scatterplot(data=x1,x='minSz',y='rr50_pow',hue='history',size='minSz',style='history',legend=False,palette='viridis',alpha=0.6)
                plt.xlim([0,10])
                plt.ylim([.7,1])

            plt.title(f'N={N} DRG={DRG}')
            plt.xlabel('')

            plt.grid(True)
            
            
            plt.subplot(2,3,1+p + 3)
            N = pairsMP[p,0]
            DRG = pairsMP[p,1]
            x1 = x[np.bitwise_and(x['N']==N,x['DRG']==DRG)]
    
            doL = (p==2)
            if PCBflag==1:
                sns.scatterplot(data=x1,x='PCB_MPC',y='mpc_pow',hue='minSz',size='minSz',style='history',legend=doL,palette='viridis',alpha=0.6)
                plt.xlim([0,40])    
                plt.ylim([.7,1])    
            elif PCBflag==0:
                ax = sns.scatterplot(data=x1,x='minSz',y='PCB_MPC',hue='minSz',size='minSz',style='history',legend=doL,palette='viridis',alpha=0.6)
                plt.xlim([0, 10])
                plt.ylim([0,40])
            elif PCBflag==2:
                sns.scatterplot(data=x1,x='diffM',y='mpc_pow',hue='minSz',size='minSz',style='history',legend=doL,palette='viridis',alpha=0.6)
                plt.xlim([0,40])
                plt.ylim([.7,1])
            elif PCBflag==3:
                sns.scatterplot(data=x1,x='minSz',y='mpc_pow',hue='history',size='minSz',style='history',legend=doL,palette='viridis',alpha=0.6)
                plt.xlim([0,10])
                plt.ylim([.7,1])
                
            if doL:
                plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            plt.title(f'N={N} DRG={DRG}')
            plt.grid(True)

        plt.show()
        
        
def define_the_shape_of_power():
    pcbLIST = np.linspace(0,.40,41)
    diffLIST = np.linspace(0,.16,33)
    N = 200
    rr50 = np.zeros((len(pcbLIST),len(diffLIST)))
    df = pd.DataFrame()
    for pcbi,PC_pcb in enumerate(pcbLIST):
        for diffi,diff_PC in enumerate(diffLIST):
            tf =calc_basic_fisherTF(N,PC_pcb,diff_PC)            
            df1 = pd.DataFrame({'TF':[tf],'PCB':PC_pcb,'diff':diff_PC})
            df = pd.concat([df,df1])            
    
    df = df.pivot("diff", "PCB", "TF")
    #sns.heatmap(data=df,annot=False,fmt='d',linewidths=0.5,vmin=0,vmax=1)
    sns.heatmap(data=df,annot=False,fmt='d',linewidths=0.5)
    
    plt.xlabel('placebo')
    plt.ylabel('diff')


def calc_basic_fisherTF(N,PCB,diffP):
    halfN = np.round(N/2)
    num_placebo_arm_responders     = np.round(halfN*PCB)
    num_drug_arm_responders        = np.round(halfN*(PCB+diffP))
    num_placebo_arm_non_responders = halfN - num_placebo_arm_responders
    num_drug_arm_non_responders    = halfN   - num_drug_arm_responders

    table = np.array([[num_placebo_arm_responders, num_placebo_arm_non_responders], [num_drug_arm_responders, num_drug_arm_non_responders]])

    [_, RR50_p_value] = stats.fisher_exact(table)
    temp=int(RR50_p_value<0.05)
    return temp
    #temp = (RR50_p_value>=0.05)*(1-RR50_p_value) + (RR50_p_value<0.05)*1
    return temp


# new idea
def tryComparisonRTM(numCPUs=9,reps=10000,DRG=0):
    Z = np.zeros((9,18))
    for Cr in range(9):
        with Parallel(n_jobs=numCPUs, verbose=False) as par:
            temp = par(delayed(inner_loopA)(Cr,DRG) for _ in trange(reps))

        X = np.array(temp,dtype=float)
        y = np.mean(X,axis=0)
        Z[Cr,:] = y.transpose()
    
    D = pd.DataFrame(Z,columns=['mT1','mT3','mT6','mT12','mF1','mF3','mF6','mF12','M','rT1','rT3','rT6','rT12','rF1','rF3','rF6','rF12','R'],
                     index=[0,1,2,3,4,5,6,7,8])
    return D

def inner_loopA(Cr,DRG):
    baseline_dur = 2    # this is an assumed constant for now
    test_dur = 3        # this is an assumed constant for now
    rct_dur = baseline_dur + test_dur
    number_of_months = 240
    number_of_days = 30*number_of_months       
    sampRATE =1
    RTMtypes = np.zeros(6)
    seizure_diary_final,mSF,overdispersion,rate,modulated_rate,theFreqs,theAmps,cycles,do_clusters,modulated_cluster_rate,seizure_diary_A,seizure_diary_B = simulator_base(sampRATE,number_of_days, returnDetails = True)
    monthly_diary = downsample(seizure_diary_final,30)
    mean_rate = np.mean(monthly_diary)
    
    # make X fake trials (entirely based on natural variability)
    number_of_trials = number_of_months - rct_dur    
    B = np.zeros(number_of_trials)
    T = np.zeros(number_of_trials)
    for k in range(number_of_trials):
        thisSet = monthly_diary[k:(k+rct_dur)]
        B[k] = np.mean(thisSet[0:baseline_dur])
        T[k] = np.mean(thisSet[baseline_dur:]) * (1-DRG)
    
    PC = 100*np.divide(B-T,B,out=np.zeros(number_of_trials), where=B!=0)
    mPC = np.median(PC)
    rPC = 100*np.mean(PC>=50)
    
    mPCsub = np.zeros((2,4))
    rPCsub = np.zeros((2,4))
    for bi,baseTF in enumerate([True,False]):
        for hi,h in enumerate([1,3,6,12]):
            availableTries = 24    
            # loop through the first couple years
            
            useMe = np.zeros(number_of_trials)
            for k2 in range(availableTries):
                startI1 = k2
                startI2 = startI1 + h
                startI3 = startI2 + baseline_dur
                histQual  = np.mean(B[startI1:startI2]) >= Cr
                baselineQual =  np.mean(B[startI2:startI3]) >= Cr
                if baseTF:
                    useMe[startI2] = 0+(histQual and baselineQual)
                else:
                    useMe[startI2] = 0+(histQual)
            PCsub = PC[useMe==1]
            if len(PCsub)>0:
                mPCsub[bi,hi] = np.median(PCsub)
                rPCsub[bi,hi] = 100*np.mean(PCsub>=50)
    X = np.zeros(18)
    X[0:4] = mPCsub[0,:]
    X[4:8] = mPCsub[1,:]
    X[8] = mPC
    X[9:13] = rPCsub[0,:]
    X[13:17] = rPCsub[1,:]
    X[17] = rPC
    return X

## next new idea

def findRTMfrac(numCPUs=9,reps=10000):
    Z = pd.DataFrame()
    for Cr in range(9):
        for hi,hist in enumerate([1,3,6,12]):
            for bi,baseTF in enumerate([True,False]):
                with Parallel(n_jobs=numCPUs, verbose=False) as par:
                    temp = par(delayed(inner_loopB)(hist,baseTF,Cr) for _ in trange(reps))
                X = np.array(temp,dtype=float)
                tempZ = pd.DataFrame({'Cr':[Cr],'hist':[hist],'baseTF':[baseTF],'fracRTM':[np.nanmedian(X)]})
                Z = pd.concat([Z,tempZ])

        print(Z)    
    return Z

def inner_loopB(hist,baseTF,Cr):
    baseline_dur = 2    # this is an assumed constant for now
    test_dur = 3        # this is an assumed constant for now
    rct_dur = baseline_dur + test_dur
    number_of_months = 12*30
    number_of_days = 30*number_of_months       
    sampRATE =1
    seizure_diary_final,mSF,overdispersion,rate,modulated_rate,theFreqs,theAmps,cycles,do_clusters,modulated_cluster_rate,seizure_diary_A,seizure_diary_B = simulator_base(sampRATE,number_of_days, returnDetails = True)
    monthly_diary = downsample(seizure_diary_final,30)
    mean_rate = np.mean(monthly_diary)

    if baseTF == True:
        theBaseBox = np.ones(baseline_dur)
    else:
        theBaseBox = np.zeros(baseline_dur)
    
    boxE = np.concatenate([np.ones(hist),theBaseBox,np.zeros(test_dur)])
    boxB = np.concatenate([np.zeros(hist),np.ones(baseline_dur),np.zeros(test_dur)])
    boxT = np.concatenate([np.zeros(hist),np.zeros(baseline_dur),np.ones(test_dur)])
    Elist = np.convolve(monthly_diary,boxE / np.sum(boxE),'valid')
    Blist = np.convolve(monthly_diary,boxB / np.sum(boxB),'valid')
    Tlist = np.convolve(monthly_diary,boxT / np.sum(boxT),'valid')
    RTM = np.logical_and(Blist>mean_rate,(np.abs(Blist-mean_rate) > np.abs(Tlist-mean_rate)))
    RTMfrac = np.mean(RTM[Elist>=Cr])

    return RTMfrac

def try_one_at_a_time(SF,hist,baseTF,Cr):
    baseline_dur = 2    # this is an assumed constant for now
    test_dur = 3        # this is an assumed constant for now
    rct_dur = baseline_dur + test_dur
    number_of_months = 12*30
    number_of_days = 30*number_of_months       
    sampRATE =1
    seizure_diary_final,mSF,overdispersion,rate,modulated_rate,theFreqs,theAmps,cycles,do_clusters,modulated_cluster_rate,seizure_diary_A,seizure_diary_B = simulator_base(sampRATE,number_of_days, defaultSeizureFreq=SF,returnDetails = True)
    monthly_diary = downsample(seizure_diary_final,30)
    mean_rate = np.mean(monthly_diary)

    if baseTF == True:
        theBaseBox = np.ones(baseline_dur)
    else:
        theBaseBox = np.zeros(baseline_dur)
    
    boxE = np.concatenate([np.ones(hist),theBaseBox,np.zeros(test_dur)])
    boxB = np.concatenate([np.zeros(hist),np.ones(baseline_dur),np.zeros(test_dur)])
    boxT = np.concatenate([np.zeros(hist),np.zeros(baseline_dur),np.ones(test_dur)])
    Elist = np.convolve(monthly_diary,boxE / np.sum(boxE),'valid')
    Blist = np.convolve(monthly_diary,boxB / np.sum(boxB),'valid')
    Tlist = np.convolve(monthly_diary,boxT / np.sum(boxT),'valid')
    RTM = np.logical_and(Blist>mean_rate,(np.abs(Blist-mean_rate) > np.abs(Tlist-mean_rate)))
    RTMfrac = np.mean(RTM[Elist>=Cr])

    return RTMfrac

def tryOut_fracs(SF,Cr,baseTF=True,hist=1,reps=5000,numCPUs = 9):
    #SF = 3
    #hist = 1
    #baseTF = True
    #Cr = 4
    #reps = 5000
    
    with Parallel(n_jobs=numCPUs, verbose=False) as par:
        temp = par(delayed(try_one_at_a_time)(SF,hist,baseTF,Cr) for _ in range(reps))
    fraclist = np.array(temp,dtype=float)
    #fraclist = np.array([try_one_at_a_time(SF,hist,baseTF,Cr) for _ in range(reps)],dtype=float)

    medF = np.nanmean(fraclist)
    return medF

def build_a_set_of_fracs(fname,baseTF,hist,reps,numCPUs = 10):
    z = pd.DataFrame()
    for si,SF in tqdm(enumerate(np.linspace(1,10,10))):
        for ci,Cr in tqdm(enumerate(np.linspace(0,8,9))):
            z = pd.concat([z,pd.DataFrame({'SF':[SF],'Cr':[Cr],'Frac':tryOut_fracs(SF,Cr,baseTF,hist,reps,numCPUs)})])
    print(z)
    z.to_csv(fname, index_label=False)

def make_many_frac_sets():
    for baseTF in [True,False]:
        for hist in [1,3,6,12]:
            build_a_set_of_fracs(f'fracs_{baseTF}_{hist}.csv',baseTF,hist,reps=5000)
        
def draw_fracs():
    plt.figure(figsize=(8,6))
    plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True)
    i = 1
    for hist in [1,12]:
        for baseTF in [True,False]:
            plt.subplot(2,2,i)
            do_one_draw_frac(baseTF,hist,legendTF=(i==2))
            i+=1
            
def do_one_draw_frac(baseTF,hist,legendTF):
    fn = f'fracs_{baseTF}_{hist}.csv'
    z = pd.read_csv(fn)
    z2 = z.loc[~np.isnan(z['Frac']),:]
    z2.loc[:,'Frac'] = 100*z2.loc[:,'Frac']
    #z2.loc[:,'Frac'] = 100*z2.loc[:,'Frac']
    
    #plt.subplot(2,1,bi)
    # Create a line plot with SF on the x-axis and Frac on the y-axis
    # Use Cr as the hue to draw a line for each unique value of Cr
    sns.scatterplot(data=z2,x='SF',y='Frac',legend=legendTF)
    #clist = np.linspace(0,1,11)
    clist = ['gray','magenta','red','orange','yellow','green','cyan','blue','black']
    for c in range(9):
        ind = (z2.Cr == c)
        z3 = z2[ind].copy()
        #print(z2[ind])
        if legendTF==True:
            plt.plot(z3.SF,z3.Frac,'-x',color=clist[c],label=f'Cr={c}')
        else:
            plt.plot(z3.SF,z3.Frac,'-x',color=clist[c])
        #sns.lineplot(data=z3, x="SF", y="Frac", hue=c,markers=True)
    plt.ylim(0,100)
    if legendTF==True:
        plt.legend(title='Cr',bbox_to_anchor=(1.05, 1.05))
    plt.title(f'BaseTF:{baseTF} hist:{hist}')
    #plt.show()
    
def build_a_set_of_fracs_withALL(fname,baseTF,hist,reps):
    z = pd.DataFrame()
    SF = -1
    for ci,Cr in tqdm(enumerate(np.linspace(0,8,9))):
        z = pd.concat([z,pd.DataFrame({'Cr':[Cr],'Frac':tryOut_fracs(SF,Cr,baseTF,hist,reps)})])
    print(z)
    z.to_csv(fname, index_label=False)
    
def do_oneAll_draw_frac(baseTF,hist,legendTF):
    fn = f'fracs-ALL_{baseTF}_{hist}.csv'
    z = pd.read_csv(fn)
    z2 = z.loc[~np.isnan(z['Frac']),:]
    z2.loc[:,'Frac'] = 100*z2.loc[:,'Frac']
    #z2.loc[:,'Frac'] = 100*z2.loc[:,'Frac']
    
    #plt.subplot(2,1,bi)
    # Create a line plot with SF on the x-axis and Frac on the y-axis
    # Use Cr as the hue to draw a line for each unique value of Cr
    sns.scatterplot(data=z2,x=1,y='Frac',legend=legendTF)
    #clist = np.linspace(0,1,11)
    clist = ['gray','magenta','red','orange','yellow','green','cyan','blue','black']
    for c in range(9):
        ind = (z2.Cr == c)
        z3 = z2[ind].copy()
        #print(z2[ind])
        if legendTF==True:
            plt.plot(1,z3.Frac,'-x',color=clist[c],label=f'Cr={c}')
        else:
            plt.plot(1,z3.Frac,'-x',color=clist[c])
        #sns.lineplot(data=z3, x="SF", y="Frac", hue=c,markers=True)
    plt.ylim(0,100)
    if legendTF==True:
        plt.legend(title='Cr',bbox_to_anchor=(1.05, 1.05))
    plt.title(f'BaseTF:{baseTF} hist:{hist}')
    #plt.show()
    
def draw_all_fracs():
    plt.figure(figsize=(8,6))
    plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True)
    i = 1
    for hist in [1,12]:
        for baseTF in [True,False]:
            plt.subplot(2,2,i)
            do_oneAll_draw_frac(baseTF,hist,legendTF=(i==2))
            i+=1
            

def try_one_at_a_time_manyFracs(SF,hist,baseTF,Cr,number_of_months):
    baseline_dur = 2    # this is an assumed constant for now
    test_dur = 3        # this is an assumed constant for now
    rct_dur = baseline_dur + test_dur
    #number_of_months = 12*30
    number_of_days = 30*number_of_months       
    sampRATE =1
    seizure_diary_final,mSF,overdispersion,rate,modulated_rate,theFreqs,theAmps,cycles,do_clusters,modulated_cluster_rate,seizure_diary_A,seizure_diary_B = simulator_base(sampRATE,number_of_days, defaultSeizureFreq=SF,returnDetails = True)
    monthly_diary = downsample(seizure_diary_final,30)
    mean_rate = np.mean(monthly_diary)

    if baseTF == True:
        theBaseBox = np.ones(baseline_dur)
    else:
        theBaseBox = np.zeros(baseline_dur)
    
    boxE = np.concatenate([np.ones(hist),theBaseBox,np.zeros(test_dur)])
    boxB = np.concatenate([np.zeros(hist),np.ones(baseline_dur),np.zeros(test_dur)])
    boxT = np.concatenate([np.zeros(hist),np.zeros(baseline_dur),np.ones(test_dur)])
    Elist = np.convolve(monthly_diary,boxE / np.sum(boxE),'valid')
    Blist = np.convolve(monthly_diary,boxB / np.sum(boxB),'valid')
    Tlist = np.convolve(monthly_diary,boxT / np.sum(boxT),'valid')
    Bdiff = (Blist - mean_rate)
    Tdiff = (Tlist - mean_rate)
    B_up = (Bdiff>0)
    B_down = np.logical_not(B_up)
    Regressing = (np.abs(Bdiff) > np.abs(Tdiff))
    notRegressing = np.logical_not(Regressing)
    GoingMoreDown = np.logical_and(notRegressing,Tdiff<0)
    GoingMoreUp = np.logical_and(notRegressing,Tdiff>0)
    improving_RTM = np.logical_and(B_up,Regressing)
    improving_upAndWayDown = np.logical_and(B_up,GoingMoreDown)
    improving_downAndWayDown = np.logical_and(B_down,GoingMoreDown)
    
    worsening_RTM = np.logical_and(B_down,Regressing)
    worsening_downAndWayUp = np.logical_and(B_down,GoingMoreUp)
    worsening_upAndWayUp = np.logical_and(B_up,GoingMoreUp)
    
    eligible_times = Elist>=Cr
    PC = 100*np.divide(Blist-Tlist,Blist)
    RTMfrac = np.zeros(7)
    RTMfrac[0] = np.nanmean(improving_RTM[eligible_times])
    RTMfrac[1] = np.nanmean(improving_upAndWayDown[eligible_times])
    RTMfrac[2] = np.nanmean(improving_downAndWayDown[eligible_times])
    RTMfrac[3] = np.nanmean(worsening_RTM[eligible_times])
    RTMfrac[4] = np.nanmean(worsening_downAndWayUp[eligible_times])
    RTMfrac[5] = np.nanmean(worsening_upAndWayUp[eligible_times])
    RTMfrac[6] = np.nanmean(PC[np.logical_and(eligible_times,~np.isinf(PC))])
    return RTMfrac

def build_a_set_of_fracs():
    reps = 5000
    number_of_months=12*30
    for counter in [1,3]:
        if counter==1:
            hist=1
            baseTF = True
        else:
            hist=3
            baseTF = False
            
        fn = f'fracsSet_{baseTF}_{hist}.csv'
        z = pd.DataFrame()
        for si,SF in tqdm(enumerate(np.linspace(1,10,10))):
            for ci,Cr in tqdm(enumerate(np.linspace(0,8,9))):
                fraclist = np.array([try_one_at_a_time_manyFracs(SF,hist,baseTF,Cr,number_of_months) for _ in range(reps)],dtype=float)
                medF = np.nanmean(fraclist,axis=0)
                medF = medF[0:6]
                zT = pd.DataFrame({'SF':[SF]*6,'Cr':[Cr]*6,'hist':[hist]*6,'baseTF':[baseTF]*6,'frac':medF,'fracInd':np.arange(6)})
                z = pd.concat([z,zT])
                #z = pd.concat([z,pd.DataFrame({'SF':[SF],'Cr':[Cr],'hist':[hist],'baseTF':[baseTF],
                #                            'RTM1':medF[0],'nRTMu1':medF[1],'nRTMd1':medF[2],
                #                            'RTM0':medF[3],'nRTMd1':medF[4],'nRTMu1':medF[5]})])
        print(fn)
        z.to_csv(fn, index=False)



def draw_all_fracSets():
    baseTFlist = [True,False]
    histlist = [1,3]
    text_str = ['Improved - RTM', 'Improved - upAndWayDown','Improved - downAndWayDown',
                'Worsened - RTM', 'Worsened - downAndWayUp','Worsened - upAndWayUp']
    z2 = pd.DataFrame()
    for i in range(2):
        hist= histlist[i]
        baseTF=baseTFlist[i]        
        fn = f'fracsSet_{baseTF}_{hist}.csv'
        z = pd.read_csv(fn)
        z['hist'] = hist
        z['baseTF'] = baseTF
        z2= pd.concat([z2,z])
    
    inds = np.array(z2['fracInd'])
    new_list = [text_str[i] for i in inds]
    z2['type'] = new_list
    sns.catplot(data=z2,kind='bar',x='SF',y='frac',hue='Cr',row='type',col='baseTF')
        
    plt.show()
    

def build_a_set_of_frac_noSF(studyChances=1,numCPUs=9,extraCalc=False):
    reps = 5000
    trialDur = 5
    number_of_months = studyChances-1 + trialDur
    for counter in [1,3]:
        if counter==1:
            hist=1
            baseTF = True
        else:
            hist=3
            baseTF = False
            
        number_of_months+=hist
        if extraCalc==True:
            # extra caluclations requested by reviewer of paper
            fn = f'fracsSetAll_extra_{number_of_months}_{baseTF}_{hist}.csv'
            fn2= f'fracsSetAll_extra_{number_of_months}_{baseTF}_{hist}_full.csv'
            Crlist = np.array([4,3,2,1.5,1,1/3,1/13])
        else:
            fn = f'fracsSetAll_{number_of_months}_{baseTF}_{hist}.csv'
            fn2= f'fracsSetAll_{number_of_months}_{baseTF}_{hist}_full.csv'
            Crlist = np.linspace(0,8,9)
        z = pd.DataFrame()
        z2 = pd.DataFrame()
        SF = -1
        
        for ci,Cr in tqdm(enumerate(Crlist)):
            with Parallel(n_jobs=numCPUs, verbose=False) as par:
                temp = par(delayed(try_one_at_a_time_manyFracs)(SF,hist,baseTF,Cr,number_of_months) for _ in range(reps))
            fraclist = np.array(temp,dtype=float)
        
            #fraclist = np.array([try_one_at_a_time_manyFracs(SF,hist,baseTF,Cr,number_of_months) for _ in range(reps)],dtype=float)
            medF = np.nanmean(fraclist,axis=0)
            PC = np.nanmedian(fraclist[:,6])
            rrFrac = np.nanmean(fraclist[:,6]>=50)
            zT = pd.DataFrame({'Cr':[Cr]*6,'hist':[hist]*6,'baseTF':[baseTF]*6,'frac':medF[0:6],'fracInd':np.arange(6),'PC':PC,'rrFrac':rrFrac})
            z = pd.concat([z,zT])
            for littleI in range(7):
                z2 =  pd.concat([z2,pd.DataFrame({'Cr':[Cr]*reps,'hist':[hist]*reps,'baseTF':[baseTF]*reps,
                                'fracInd':littleI,'PC':fraclist[:,6],'rrFrac':fraclist[:,6]>50,'frac0':fraclist[:,0]})])
            
            #z = pd.concat([z,pd.DataFrame({'SF':[SF],'Cr':[Cr],'hist':[hist],'baseTF':[baseTF],
            #                            'RTM1':medF[0],'nRTMu1':medF[1],'nRTMd1':medF[2],
            #                            'RTM0':medF[3],'nRTMd1':medF[4],'nRTMu1':medF[5]})])
        print(fn)
        z.to_csv(fn, index=False)
        z2.to_csv(fn2,index=False)

def go_read_pow_data(studyChances,fnOUT,extraCalc=False):
    trialDur = 5
    baseTFlist = [True,False]
    histlist = [1,3]
    reps = 1000
    df = pd.DataFrame()
    if extraCalc==True:
        filePre = 'RCT_eff_extra'
        Crlist = np.array([4,3,2,1.5,1,1/3,1/13])
    else:
        filePre = 'RCT_eff'
        Crlist = np.linspace(0,8,9)
        
    for bi,baseTF in tqdm(enumerate(baseTFlist)):
        hist = histlist[bi]
        for minSz in Crlist:
            fn = f'{filePre}_{baseTF}_{minSz}_{studyChances}_keeper.csv'
            data = np.genfromtxt(fn, delimiter=',', skip_header=1)
            temp = pd.DataFrame({'rr50_pow':data[:,0],'mpc_pow':data[:,1]})
            for _ in range(reps):
                sample = temp.sample(n=1000, replace=True, axis=0)
                samp2 = pd.DataFrame({'rr50_pow':[np.mean(sample.rr50_pow)],
                                      'mpc_pow':[np.mean(sample.mpc_pow)],
                                      'minSz':[minSz],
                                      'baseTF':[baseTF]})
                df = pd.concat([df,samp2])
    df.to_csv(fnOUT,index=False)
                
def draw_all_fracSets_noSF(showFull=True,studyChances=1,with_error=False,extraCalc=False,fnEff=''):
    if extraCalc==False:
        fnPre = 'RCT_eff'
    else:
        fnPre = 'RCT_eff_extra'
    
    if fnEff=='':
        fnEff=f'{fnPre}_{studyChances}.csv'
    
    trialDur = 5
    number_of_months = studyChances-1 + trialDur
    baseTFlist = [True,False]
    histlist = [1,3]
    text_str = ['Improved - RTM', 'Improved - upAndWayDown','Improved - downAndWayDown',
                'Worsened - RTM', 'Worsened - downAndWayUp','Worsened - upAndWayUp']
    z2 = pd.DataFrame()
    z2FULL = pd.DataFrame()
    for i in range(2):
        hist= histlist[i]
        baseTF=baseTFlist[i] 
        number_of_months+=hist
        if extraCalc==False:
            fn = f'fracsSetAll_{number_of_months}_{baseTF}_{hist}.csv'
        else:
            fn = f'fracsSetAll_extra_{number_of_months}_{baseTF}_{hist}.csv'
        z = pd.read_csv(fn)
        z['hist'] = hist
        z['baseTF'] = baseTF
        z2= pd.concat([z2,z])
        if with_error==True:
            if extraCalc==False:
                fn2 = f'fracsSetAll_{number_of_months}_{baseTF}_{hist}_full.csv'
            else:
                fn2 = f'fracsSetAll_extra_{number_of_months}_{baseTF}_{hist}_full.csv'
            zFULL = pd.read_csv(fn2)
            zFULL['hist'] = hist
            zFULL['baseTF'] = baseTF
            z2FULL= pd.concat([z2FULL,zFULL])
            

    inds = np.array(z2['fracInd'])
    new_list = [text_str[i] for i in inds]
    z2['type'] = new_list
    z2['rrFrac'] *= 100
    z2['frac'] *=100
    z2['Baseline included'] = z2['baseTF'].apply(lambda x: 'with' if x else 'without')
    z2 = z2.rename(columns={'Cr':'Minimum rate (sz./mo.)','baseTF':'Include baseline','PC':'Median % change','frac':'Fraction of total','rrFrac':'RR50 %'})
    if with_error==True:
        #inds = np.array(z2FULL['fracInd'])
        #new_list = [text_str[i] for i in inds]
        #z2FULL['type'] = new_list
        z2FULL['rrFrac'] *= 100
        z2FULL['frac0'] *=100
        z2FULL['Baseline included'] = z2FULL['baseTF'].apply(lambda x: 'with' if x else 'without')
        z2FULL = z2FULL.rename(columns={'Cr':'Minimum rate (sz./mo.)','baseTF':'Include baseline','PC':'Median % change','frac0':'Fraction of total','rrFrac':'RR50 %'})
            
    # this is for power calculations
    df = pd.read_csv(fnEff)
    df['pow_m'] *= 100
    df['pow_r'] *= 100
    df['Baseline included'] = df['baseTF'].apply(lambda x: 'with' if x else 'without')
    df = df.rename(columns={'pow_m':'MPC Power %','pow_r':'RR50 Power %','minSz':'Minimum rate (sz./mo.)','baseTF':'Include baseline'})
    
    #sns.catplot(data=z2,kind='bar',x='Minimum rate (sz./mo.)',y='Fraction of total',hue='type',col='Include baseline',palette='rocket')
    if showFull==False:
        plt.subplots(nrows = 3, ncols=2, figsize=(8,8),sharex=True)
        plt.subplot(3,2,1)
        if with_error==False:
            ax = sns.barplot(data=z2[z2['fracInd']==0],x='Minimum rate (sz./mo.)',y='Fraction of total',
                hue='Baseline included',palette=['black','gray'])
        else:
            ax = sns.barplot(data=z2FULL,x='Minimum rate (sz./mo.)',y='Fraction of total',
                hue='Baseline included',palette=['black','gray'],errorbar=('ci',95))

            
        #print(z2[z2['fracInd']==0])
        
        ax.axes.xaxis.set_ticklabels([])
        ax.legend().set_visible(False)
        ax.set(xlabel=None)
        plt.grid(True)
        plt.ylim(0,60)
        plt.title('Fraction of RTM vs eligibility')
        plt.subplot(3,2,2)
        if with_error==False:
            ax = sns.barplot(data=z2[z2['fracInd']==0],x='Minimum rate (sz./mo.)',y='Fraction of total',
                hue='Baseline included',palette=['black','gray'])
        else:
            ax = sns.barplot(data=z2FULL,x='Minimum rate (sz./mo.)',y='Fraction of total',
                hue='Baseline included',palette=['black','gray'],errorbar=('ci',95))

        ax.axes.xaxis.set_ticklabels([])
        ax.legend().set_visible(False)
        ax.set(xlabel=None)
        plt.grid(True)
        plt.ylim(0,60)
        plt.title('Fraction of RTM vs eligibility')
        plt.subplot(3,2,3)
        if with_error==False:
            ax = sns.barplot(data=z2,x='Minimum rate (sz./mo.)',y='Median % change',hue='Baseline included',
                        palette=sns.color_palette(['black','gray']))
        else:
            ax = sns.barplot(data=z2FULL,x='Minimum rate (sz./mo.)',y='Median % change',hue='Baseline included',
                        palette=sns.color_palette(['black','gray']),errorbar=('ci',95))

        ax.axes.xaxis.set_ticklabels([])
        ax.legend().set_visible(False)
        ax.set(xlabel=None)
        plt.grid(True)
        plt.ylim(-40,40)
        plt.title('MPC vs eligibility')
        plt.subplot(3,2,4)
        if with_error==False:
            ax = sns.barplot(data=z2,x='Minimum rate (sz./mo.)',y='RR50 %',hue='Baseline included',
                        palette=sns.color_palette(['black','gray']))
        else:
            ax = sns.barplot(data=z2FULL,x='Minimum rate (sz./mo.)',y='RR50 %',hue='Baseline included',
                        palette=sns.color_palette(['black','gray']),errorbar=('ci',95))

        ax.axes.xaxis.set_ticklabels([])
        ax.set(xlabel=None)
        plt.grid(True)
        plt.ylim(0,25)
        plt.title('RR50 vs eligibility')
        plt.subplot(3,2,5)
        if with_error==False:
            ax = sns.barplot(data=df,x='Minimum rate (sz./mo.)',y='MPC Power %',hue='Baseline included',
                        palette=sns.color_palette(['black','gray']))
        else:
            zPOW = pd.read_csv(f'RTM_eff_boot_keeper_{studyChances}.csv')
            zPOW['mpc_pow'] *= 100
            zPOW['rr50_pow'] *= 100
            zPOW['Baseline included'] = zPOW['baseTF'].apply(lambda x: 'with' if x else 'without')
            zPOW = zPOW.rename(columns={'mpc_pow':'MPC Power %','rr50_pow':'RR50 Power %','minSz':'Minimum rate (sz./mo.)','baseTF':'Include baseline'})

            ax = sns.barplot(data=zPOW,x='Minimum rate (sz./mo.)',y='MPC Power %',hue='Baseline included',
                        palette=sns.color_palette(['black','gray']),errorbar=('ci',95))

        print(df)
        print('done')
        ax.legend().set_visible(False)
        plt.plot([-.5, 8.5],[90,90],'k--')
        plt.title('MPC Efficacy vs. min. rate')
        plt.ylim(0,100)
        plt.grid(True)
        plt.subplot(3,2,6)
        if with_error==False:
            ax = sns.barplot(data=df,x='Minimum rate (sz./mo.)',y='RR50 Power %',hue='Baseline included',
                        palette=sns.color_palette(['black','gray']))
        else:
            ax = sns.barplot(data=zPOW,x='Minimum rate (sz./mo.)',y='RR50 Power %',hue='Baseline included',
                        palette=sns.color_palette(['black','gray']),errorbar=('ci',95))
        ax.legend().set_visible(False)
        plt.plot([-.5, 8.5],[90,90],'k--')
        plt.title('RR50 Efficacy vs. min. rate')
        plt.ylim(0,100)
        plt.grid(True)
        plt.tight_layout(pad=1)
    
    else:
        sns.catplot(data=z2,kind='bar',x='Minimum rate (sz./mo.)',y='Fraction of total',
                    hue='type',col='Include baseline',palette=['green','cyan','blue','red','purple','yellow'])
        plt.grid(True)
        plt.ylim(0,100)
        plt.show()
        sns.catplot(data=z2,kind='bar',x='Minimum rate (sz./mo.)',y='Median % change',col='Include baseline',
                    palette=sns.color_palette(['black']))
        plt.grid(True)
        plt.ylim(-40,40)
        plt.show()
        sns.catplot(data=z2,kind='bar',x='Minimum rate (sz./mo.)',y='RR50 %',col='Include baseline',
                    palette=sns.color_palette(['black']))
        plt.grid(True)
        plt.ylim(0,30)
        
        plt.show()

    
def make_RCT_efficiency_check(fn,howLong=1,saveFile=False,extraCalc=False):
    reps=5000
    DRG=0.2
    N=400
    numCPUs=9
    PCB=0
    years=howLong/12
    
    if extraCalc == True:
        minSzList = [1/13,1/3,1,1.5,2,3,4]
        prefixTxt = 'RCT_eff'
    else:
        minSzList = [0,1,2,3,4,5,6,7,8]
        prefixTxt = 'RCT_eff_extra'
        
    df = pd.DataFrame()
    for baseTF in [True,False]:
        for minSz in minSzList:
            if baseTF==True:
                thisHist = 1
            else:
                thisHist = 3
            
            saveFn = f'{prefixTxt}_{baseTF}_{minSz}_{howLong}_keeper.csv'
            print(f'{PCB} {thisHist}')
            rr50_pow,mpc_pow,diffR,diffM,PCB_R,PCB_M = runSet_with_HandB(howMany=reps,hist=thisHist,baseTF=baseTF,minSz=minSz,N=N,years=years,DRG=DRG,PCB=PCB,
                    numCPUs=numCPUs,doPIX=False,printTF=True,saveFile=saveFile,saveFn=saveFn)
            tempd = pd.DataFrame({'N':[N],'minSz':[minSz],'DRG':[DRG],'PCB':[PCB],'history':[thisHist],'baseTF':[baseTF],'diffR':[diffR],'diffM':[diffM],'PCB_R':[PCB_R],'PCB_M':[PCB_M],'pow_r':[rr50_pow],'pow_m':[mpc_pow]})
            df = pd.concat([df,tempd])
  
    print(df)
    df.to_csv(fn,index=False)


def draw_efficacy_for_many_Min():
    df = pd.read_csv('Low-high-testv5.csv')

    df['pow_m'] *= 100
    df['pow_r'] *= 100
    df = df.rename(columns={'pow_m':'MPC Power %','pow_r':'RR50 Power %','minSz':'Minimum rate','baseTF':'Include baseline'})

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    sns.scatterplot(data=df,x='Minimum rate',y='MPC Power %',hue='Include baseline',style='Include baseline')
    plt.title('MPC Efficacy vs. Minimum seizure rate')
    plt.ylim(0,100)
    plt.grid(True)
    plt.subplot(1,2,2)
    sns.scatterplot(data=df,x='Minimum rate',y='RR50 Power %',hue='Include baseline',style='Include baseline')
    plt.title('RR50 Efficacy vs. Minimum seizure rate')
    plt.ylim(0,100)
    plt.grid(True)

def build_minSz_findN_pow(minSz,fn):
    reps=5000
    DRG=0.2
    numCPUs=9
    PCB=0
    #minSz = 2
    # the test here is which of these values of N achieves 90% power
    df = pd.DataFrame()
    for baseTF in [True,False]:
        if baseTF==True:
            thisHist = 1
        else:
            thisHist = 3
        
        if minSz<1.5:
            Nlist = [450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500]
        else:
            Nlist = [350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200]
        for N in Nlist:
            print(f'{baseTF} {N}')            
            rr50_pow,mpc_pow,diffR,diffM,PCB_R,PCB_M = runSet_with_HandB(howMany=reps,hist=thisHist,baseTF=baseTF,minSz=minSz,N=N,years=3,DRG=DRG,PCB=PCB,
                    numCPUs=numCPUs,doPIX=False,printTF=True)
            tempd = pd.DataFrame({'N':[N],'minSz':[minSz],'DRG':[DRG],'PCB':[PCB],'history':[thisHist],'baseTF':[baseTF],'diffR':[diffR],'diffM':[diffM],'PCB_R':[PCB_R],'PCB_M':[PCB_M],'pow_r':[rr50_pow],'pow_m':[mpc_pow]})
            df = pd.concat([df,tempd])
            if rr50_pow>.9:
                break
  
    print(df)
    df.to_csv(fn,index=False)
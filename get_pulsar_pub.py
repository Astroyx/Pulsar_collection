import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import copy
import sqlite3
import numpy as np
import sys
import argparse
import os
from astropy.time import Time
import matplotlib.gridspec as gridspec
from lmfit import Model
import matplotlib.patches as patches
def dedisperse(dat, dm, f0, f1, nchan, tsamp):
    # f0(i=0), f1(i=-1) in MHz, tsamp in ms
    fh = max(f0, f1)
    DM_con = 4.1488064239  #Kulkarni 2020
    freqs = np.linspace(f0, f1, nchan, endpoint=True)
    delays = np.round(DM_con * dm * (freqs**(-2.) - fh**(-2.)) / tsamp).astype(int) 
    for i, ds in enumerate(delays):
        dat[i, :] = np.roll(dat[i, :], -ds )

def unpack_data (data, data_unpack, nbits, nchan, npol, index):
    dat = data[index,npol,:]
    data = np.reshape(np.unpackbits(dat),(nchan,nbits))
    data = np.packbits(data,axis=1,bitorder='little').T
    data_unpack[index, npol, :] = np.squeeze(data)

def bin_data(dat, num_bin, num_chn):
    dat = dat.reshape(-1, num_chn, dat.shape[1]).sum(axis=1)
    dat = dat.reshape(dat.shape[0], -1, num_bin).sum(axis=2)
    return dat

def gaussian(x1, amp, cen, wid, mean):
    return amp * np.exp(-(x1-cen)**2 / (2*wid**2)) + mean 
gmodel = Model(gaussian)
def zap_rfi_freq (dat, nchan):
    ######### Remove strong narrow band RFI ############
    spec = np.sum(dat, axis=1)
    std = np.std(dat, axis=1)
    n0 = 0
    n1 = 1

    mask = np.ones(nchan, dtype=bool)
    while n0 != n1:
        n0 = nchan - np.count_nonzero(mask)
        std_spec = np.std(spec[mask])
        mean_spec = np.mean(spec[mask])

        std_std = np.std(std[mask])
        mean_std = np.mean(std[mask])
        for i in range(nchan):
            if (np.abs(spec[i]-mean_spec) > 3*std_spec) or (np.abs(std[i]-mean_std) > 3*std_std):
                mask[i] = False

        n1 = nchan - np.count_nonzero(mask)
    dat_mean = np.mean(dat[mask,:])
    obsmean=np.mean(dat)
    dat_plot=dat
    m, nbin = dat.shape
    newchan=0
    maskchan=[]
    for i in range(nchan):
        if mask[i] == False:
            dat_plot[i,:] = dat_mean
            dat[i,:] = 0
            newchan+=1
            maskchan.append(i)
    print("dat_mean",dat_mean,obsmean)
    return (nchan-newchan)/nchan,dat_plot,maskchan,dat

def fit_plot(I,mjd,bin_num,filename,idx,gain_factor=1,beam=1,t_samp=0.0001,T_sys=21,bandwidth=288*10**6):
    if beam==1:
        gain=0.735*gain_factor
    elif beam<8:
        gain=0.690*gain_factor
    else:
        gain=0.581*gain_factor

    start=int(0.3*len(I))
    end=int(0.7*len(I))
    nbin=np.arange(len(I))
    y_background=I
    y_background=np.delete(y_background,[i for i in range(start,end)],axis=0)
    raw_background=np.mean(y_background)
    I=I-raw_background
    y_background=y_background-raw_background
    x_background=np.delete(nbin,[i for i in range(start,end)],axis=0)    
    params = np.polyfit(x_background, y_background, 0)
    fit_y = np.polyval(params, nbin)  

    newI=I-fit_y
    newy_background=newI
    newy_background=np.delete(newy_background,[i for i in range(start,end)],axis=0)
    poly_background=np.mean(newy_background)  #background level after polynomial fit

    yfit=newI[start:end]
    RMS = np.std(newy_background)
    x = np.arange(0,len(yfit),1)
    amp = np.max(yfit)
    cen = np.mean(np.where(yfit==amp))
    fitwidth=w50/1000/t_samp/bs/3
    fit_result=""
    for i in range(2):
        result = gmodel.fit(yfit, x1=x, amp=amp, cen=cen, wid=fitwidth, mean=0)
        wid_fit = abs(result.params['wid'].value)
        cen_fit = result.params['cen'].value

        mean_fit = result.params['mean'].value
        b_s = max(round(cen_fit-wid_fit*3),0) # burst start index
        b_e = min(round(cen_fit+wid_fit*3),len(yfit)-1) # burst end index
        y_burst = yfit[b_s:b_e+1]
        if len(y_burst)==0:
            cen=len(yfit)/2
            fitwidth=1
    if b_s>b_e or abs(mean_fit)>1.5*RMS or cen_fit<0 or cen_fit>(end-start):
        y_burst = yfit
        mean_fit=poly_background
        cen_fit=len(I)//2-start
        b_s,b_e=start,end-1
        fit_result="(failed)"

    t_samp=t_samp*bin_num
    time_depart=(cen_fit+start-len(I)//2+idx)*t_samp/3600/24
    y_burst = y_burst-mean_fit
    SNR=np.max(y_burst)/RMS
    FWHM=abs(2.355*wid_fit)      
    wid_fit_w50 = FWHM * t_samp * 1000 # ms, width of the burst from gaussian fit
    flux = (np.max(y_burst)/RMS)*T_sys/gain/(2*bandwidth*t_samp)**0.5 * 1000  #mJy
    width_effective = np.sum(y_burst)/np.max(y_burst) * t_samp * 1000  #ms, width of the burst from the fluence
    fluence = flux * width_effective / 1000  #Jyms
    if width_effective<0.1 or fluence<0.4:
        time_depart=0
        y_burst = I[start:end]
        b_s,b_e=start,end-1
        SNR=np.max(y_burst)/RMS
        flux = (np.max(y_burst)/RMS)*T_sys/gain/(2*bandwidth*t_samp)**0.5 * 1000  #mJy
        width_effective = np.sum(y_burst)/np.max(y_burst) * t_samp * 1000  #ms
        fluence = flux * width_effective / 1000  #Jyms
        fit_result="(failed)"
    if not os.path.exists(dirname+"/fitpng"):
        os.mkdir(dirname+"/fitpng")
    figg = plt.figure(figsize=(12,10),facecolor='w')
    figg_plot = figg.add_subplot(311)
    figg_plot.plot(x+start,yfit+0.15*np.max(I)+raw_background,color='g')
    figg_plot.plot(I+raw_background,color='black',label = "data with baseline")
    plt.legend()
    figg_plot = figg.add_subplot(312)
    figg_plot.plot(x+start,yfit+0.3*np.max(newI)+poly_background,color='g',label = str(mjd))
    figg_plot.plot(newI,color='black',label = "data without baseline")
    plt.legend()
    figg_plot = figg.add_subplot(313) 
    plt.plot(x,yfit,'g-',label = 'fit region')
    plt.plot(np.arange(b_s,b_e+1,1),y_burst,'b-', label = 'burst region')
    gaussian_label='gaussian fitted profile'+fit_result
    plt.plot(x, result.best_fit, 'r--', label = gaussian_label)
    plt.legend()
    figg.savefig(dirname+"/fitpng/"+filename+".png",dpi=100)
    plt.close() 
    
    return fluence,flux,width_effective,SNR,wid_fit_w50,time_depart,RMS
def takeZero(elem):
    return elem[0]

parser = argparse.ArgumentParser(description='Read PSRFITS format search mode data')
parser.add_argument('-bf',  '--bin_chn', default=1, type=int, help='Bin the data in freq by a factor of bf.')
parser.add_argument('-bs',  '--bin_samp', default=1, type=int, help='Bin the data in time by a factor of bs.')
parser.add_argument('-jlist',  '--jlist',  action='store_true', help='List all the pulsar name in the database')
parser.add_argument('-fluence',  '--fit_fluence',  action='store_true', help='Fit the fluence of the pulsar')
parser.add_argument('-j',  '--input_jname',default="J0758-1528",  type=str, help='Input Jname')
parser.add_argument('-db',  '--database', default='Pulsar_fits_database_v1.db', type=str, help='Input database location')
parser.add_argument('-mjd',  '--MJD', default='all', type=str, help='Input needed MJD')
args = parser.parse_args()
databasename=args.database
aonn = sqlite3.connect(databasename)
mycursor = aonn.cursor()
myjname=args.input_jname
bf = args.bin_chn
bs = args.bin_samp
needmjd = args.MJD
jlist=args.jlist
fit_fluence=args.fit_fluence
###########################LIST ALL JNAME AND EPOCH############################
if jlist:
    aa=mycursor.execute('SELECT pulsarID,jname,dm,p0,w50 FROM pulsar')
    print("pulsarID,Jname,dm,p0,w50,MJD_obs")
    a= [[] for _ in range(6)]
    for row in aa:
        for i in range(5):
            a[i].append(row[i])
    for i in range(len(a[0])):
        bb=mycursor.execute('SELECT timeStartMJD FROM file WHERE pulsarID = ?', (a[0][i],))
        mjdlist=[]
        for row in bb:
            mjdlist.append(int(row[0]))
        a[5].append(np.unique(mjdlist))
        print(a[0][i],a[1][i],a[2][i],a[3][i],a[4][i],a[5][i])
    sys.exit()
mycursor.execute('SELECT pulsarID,jname,dm,p0,w50,pepoch FROM pulsar WHERE jname = ?', (myjname,))
pid,jname,dm,p0,w50,pepoch = mycursor.fetchone()
print("pid,jname,dm,p0,pepoch,w50",pid,jname,dm,p0,pepoch,w50)
if w50==None or w50=="nan" or w50=="*":
    w50=1.0
dirname=str(myjname)
if not os.path.exists(dirname):
    os.mkdir(dirname)


aa=mycursor.execute('SELECT filesegName,data,timeStartMJD,pulseNumber,dist_d,gain_factor FROM fileSegment JOIN file ON fileSegment.pfLinkID=file.pfLinkID WHERE fileSegment.pulsarID = ?', (pid,))
###########################EXTRACT PSRFITS DATA############################
for row in aa:
    if str(int(row[2]))==needmjd or needmjd=="all":
        
        with open(dirname+"/"+row[0], 'wb') as f:
            f.write(row[1])
        in_file = dirname+"/"+row[0]
        print(dirname,row[0])
###########################################################################
aa=mycursor.execute('SELECT filesegName,data,timeStartMJD,pulseNumber,dist_d,gain_factor FROM fileSegment JOIN file ON fileSegment.pfLinkID=file.pfLinkID WHERE fileSegment.pulsarID = ?', (pid,))
###########################GET FLUENCE#####################################
if fit_fluence:
    compress=open(dirname+"_obs"+str(needmjd)+'_parameter.txt',mode='w',newline='\n')
    compress.write("Filename MJD flux(mJy),fluence(Jyms),width_eff(ms),RMS,tsamp\n")
    allfluence=[]
    allmjd=[]
    allrms=[]
    segname=[]
    for row in aa:
        if str(int(row[2]))==needmjd or needmjd=="all":
            in_file = dirname+"/"+row[0]
            print(dirname,row[0])
            ###########################READ PSRFITS DATA############################
            hdulist = fits.open(in_file)
            tbdata = hdulist['SUBINT'].data
            obsbw = hdulist['PRIMARY'].header['OBSBW']
            nbits = hdulist['SUBINT'].header['NBITS']
            nchan = int(hdulist['SUBINT'].header['NCHAN'])
            nsblk = int(hdulist['SUBINT'].header['NSBLK'])
            nsub = hdulist['SUBINT'].header['NAXIS2']   #nsub=1 in all file segment
            tsamp = float(hdulist['SUBINT'].header['TBIN'])
            cFreq = float(hdulist[0].header['OBSFREQ'])   # MHz
            npol = hdulist['SUBINT'].header['NPOL']
            nstot=hdulist['SUBINT'].header['NSTOT']
            beamid=int(hdulist['PRIMARY'].header["IBEAM"])
            imjd=hdulist['PRIMARY'].header['STT_IMJD']
            smjd=hdulist['PRIMARY'].header['STT_SMJD']
            tmjd=hdulist['PRIMARY'].header['STT_OFFS']
            realmjd=imjd+smjd/3600./24.+tmjd/3600./24.
            mjdt = Time(realmjd, format='mjd')
            obstime = mjdt.iso
            dat_freq = tbdata['DAT_FREQ']
            f0 = dat_freq[0][0]
            f1 = dat_freq[0][-1]
            print(nsblk)
            ###########################UNPACK DATA############################
            data_unpack = np.empty((nsblk, npol, nchan), dtype=int)
            data = np.reshape(tbdata['DATA'], (nsub*nsblk, npol, int(nchan/(8/nbits))))
            for i in range(nsblk):
                for j in range(npol):
                    unpack_data (data, data_unpack, nbits, nchan, j, i)
            data_out = np.moveaxis(data_unpack, 0, -1)
            if npol != 1:
                data_out = 0.5*(data_out[0, :, :]+data_out[1, :, :])
            else:
                data_out = np.squeeze(data_out)
            newchan_factor,dat_plot,maskchan,data_out=zap_rfi_freq (data_out, nchan)
            print("newchan_factor",newchan_factor)
            data_dedisp = copy.deepcopy(data_out)
            data_dedisp2=copy.deepcopy(dat_plot)
            dedisperse(data_dedisp,dm,f0/1000,f1/1000,nchan,tsamp*1000)
            dedisperse(data_dedisp2,dm,f0/1000,f1/1000,nchan,tsamp*1000)
            data_dedisp=bin_data (data_dedisp, bs, bf)
            data_dedisp2=bin_data (data_dedisp2, bs, bf)
            data_out=bin_data (data_out, bs, bf)
            integral_line=np.sum(data_dedisp, axis=0)/(nchan-len(maskchan))

            binnum=int(w50/1000/tsamp/bs/10)+1
            print("binnum",binnum)
            m = len(integral_line)
            weight = np.arange(0, m, binnum)
            binI = np.add.reduceat(integral_line, weight)
            idx=np.where(binI==np.max(binI))[0][0]*binnum
            print(idx)
            realfilename=row[0].split(".sf")[0]
            ###########################GET FLUENCE############################
            gain_factor=float(row[5])
            if row[3]>1:
                slice_number=int(nsblk*tsamp/p0)
                slice_nsblk=int(nsblk/slice_number)
                for q in range(slice_number):
                    slice_integral_line=integral_line[q*slice_nsblk:(q+1)*slice_nsblk]
                    binnum=int(w50/1000/tsamp/bs/5)+1
                    m = len(slice_integral_line)
                    weight = np.arange(0, m, binnum)
                    binI = np.add.reduceat(slice_integral_line, weight)
                    tempidx=np.where(binI==np.max(binI))[0][0]*binnum

                    slice_integral_line = np.roll(slice_integral_line, -tempidx+slice_nsblk//2) #find the peak,mv to the center of the bin 
                    slice_idx=tempidx-q*slice_nsblk
                    
                    slice_filename=realfilename+"_"+str(q)
                    fluence,flux,width_effective,SNR,wid_fit_w50,time_depart,RMS=fit_plot(slice_integral_line,realmjd,bs,slice_filename,slice_idx,gain_factor=gain_factor,beam=beamid,t_samp=tsamp,T_sys=21,bandwidth=obsbw*10**6*newchan_factor)
                    print("fluence",fluence)
                    if SNR>2 and fluence>0.2:
                        segname.append('_'.join(row[0].strip('\n').split("_")[0:-5])+".sf")
                        allrms.append(RMS)
                        allfluence.append(fluence)
                        allmjd.append(realmjd+time_depart)
                        compress.write("%s %.12f %.2f %.2f %.2f %.2f %f\n"%(realfilename,realmjd+time_depart,flux,fluence,width_effective,RMS,tsamp*bs))
            elif row[3]==1:

                integral_line = np.roll(integral_line, -idx+len(integral_line)//2)
                fluence,flux,width_effective,SNR,wid_fit_w50,time_depart,RMS=fit_plot(integral_line,realmjd,bs,realfilename,idx,gain_factor=gain_factor,beam=beamid,t_samp=tsamp,T_sys=21,bandwidth=obsbw*10**6*newchan_factor)
                print("fluence",fluence)
                if fluence>0.2:
                    segname.append('_'.join(row[0].strip('\n').split("_")[0:-5])+".sf")
                    allrms.append(RMS)
                    allfluence.append(fluence)
                    allmjd.append(realmjd+time_depart)
                    compress.write("%s %.12f %.2f %.2f %.2f %.2f %f\n"%(realfilename,realmjd+time_depart,flux,fluence,width_effective,RMS,tsamp*bs))

            ###########################PLOT DYNAMIC SPECTRUM############################
            data_dedisp2=np.roll(data_dedisp2, -idx+len(integral_line)//2,axis=1)  #move the peak to the center
            gs = gridspec.GridSpec(9,4)
            fig = plt.figure(figsize=(5.5,8),facecolor='w')
            plt.rc('font', size=14) 
            plt.subplots_adjust(wspace =0, hspace =0)
            ax =plt.subplot(gs[1:4,0:4])
            plt.imshow(dat_plot,aspect="auto")
            weighty = np.linspace(f0,f1,3,endpoint=True)
            weighty = [round(i) for i in weighty]
            weightx = np.linspace(0, nsblk*tsamp, 5,endpoint=True)
            weightx = [round(i,2) for i in weightx]
            plt.yticks(np.linspace(0,dat_plot.shape[0],3, endpoint = True),weighty)
            plt.xticks(np.linspace(0,dat_plot.shape[1],5, endpoint = True),weightx)
            plt.ylabel("Frequency(MHz)")
            plt.xlabel("Time(s)")
            for i in range(len(maskchan)):
                rect_filled = patches.Rectangle((0, maskchan[i]-0.5), data_dedisp2.shape[1]/20, 1, facecolor='red', alpha=1)
                ax.add_patch(rect_filled)
            bx =plt.subplot(gs[0:1,0:4])
            plt.plot(np.arange(dat_plot.shape[1]),np.sum(dat_plot, axis=0))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0,dat_plot.shape[1]])
            cx =plt.subplot(gs[6:9,0:4])
            plt.imshow(data_dedisp2,aspect="auto")
            weighty = np.linspace(f0,f1,3,endpoint=True)
            weighty = [round(i) for i in weighty]
            weightx = np.linspace(0, nsblk*tsamp, 5,endpoint=True)
            weightx = [round(i,2) for i in weightx]
            plt.yticks(np.linspace(0,data_dedisp2.shape[0],3, endpoint = True),weighty)
            plt.xticks(np.linspace(0,data_dedisp2.shape[1],5, endpoint = True),weightx)
            plt.ylabel("Frequency(MHz)")
            plt.xlabel("Time(s)")
            for i in range(len(maskchan)):
                rect_filled = patches.Rectangle((0, maskchan[i]-0.5), data_dedisp2.shape[1]/20, 1, facecolor='red', alpha=1)
                cx.add_patch(rect_filled)
            dx =plt.subplot(gs[5:6,0:4])
            plt.plot(np.arange(data_dedisp2.shape[1]),np.sum(data_dedisp2, axis=0))
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0,data_dedisp2.shape[1]])
            fig.savefig(dirname+"/"+row[0].split(".sf")[0]+"_"+str(realmjd+time_depart)+".png",dpi=100,bbox_inches = 'tight')
            plt.close()


    compress.close()     

    ###########################PLOT SUMMARY FIG############################
    unifilename=np.unique(segname)
    days=np.unique(np.int64(allmjd))
    filestart=[]
    fileend=[]
    dists=[]
    gain_factors=[]
    for i in range(len(unifilename)):
        aa=mycursor.execute('SELECT obs_length,timeStartMJD,dist_d,gain_factor FROM file  WHERE filename = ?', (unifilename[i],))
        for row in aa:
            filestart.append(row[1])
            dists.append(row[2])
            gain_factors.append(row[3])
            fileend.append(row[1]+row[0]/3600/24)
    part=0
    if len(days)//2>4:  #if there are more than 9 epochs, only plot the first 9
        days=days[:9]
        part=1
    print("Number of file segment: ",len(segname))
    print("Obervation epoch: ",filestart,fileend)
    plt.rc('font', size=15) 
    fig=plt.figure(figsize=(12,5*(len(days)//2+1)),facecolor='w')
    allrms=allrms/np.mean(allrms)*np.mean(allfluence)
    for i in range(len(days)):
        idx=np.where(np.int64(allmjd)==days[i])[0]
        idx2=np.where(np.int64(filestart)==days[i])[0]
        idx3=np.where(np.int64(fileend)==days[i])[0]
        idx2=np.unique(np.concatenate((idx2,idx3),axis=0))
        a= [[] for _ in range(4)]
        for s in range(len(idx)):
            a[3].append(allrms[idx[s]])
            a[2].append(allfluence[idx[s]])
            a[1].append(allmjd[idx[s]])
        startmjd=min(np.array(filestart)[idx2])
        endmjd=max(np.array(fileend)[idx2])
        totalwidth=(endmjd-startmjd)*3600*24/p0
        p0mjd=p0/3600/24
        for s in range(len(a[1])):
            a[0].append(round((a[1][s]-startmjd)*3600*24/p0))
        a=list(map(list, zip(*a)))
        a.sort(key=takeZero)     #sort the array by file start time
        a=list(map(list, zip(*a)))
        plt.subplot(len(days)//2+1,2,i+1)
        plt.bar(a[0],a[2], width=totalwidth/250,alpha=1,color="black")
        plt.plot(a[0],a[3],color="red",label="RMS level")
        #plt.plot(a[0],a[3],color="b",label="Threshold level")
        if len(a[0])==1:
            plt.errorbar(a[0],a[3],xerr=totalwidth/50,color="red")
        for s in range(len(idx2)):
            realidx=idx2[s]
            middle=(fileend[realidx]+filestart[realidx])/2.0
            obswidth=round((fileend[realidx]-filestart[realidx])*3600*24/p0)
            plt.bar((middle-startmjd)*3600*24/p0,np.mean(allfluence)/7, width=obswidth,alpha=0.75,color="tab:cyan")
        if len(np.array(dists)[idx2])==1:
            plt.xlabel(r"Epoch %d (MJD %d, $\rm D_s$ %.3f, $\rm G_f$ %.3f)"%(i+1,days[i],np.array(dists)[idx2][0],np.array(gain_factors)[idx2][0]))
        else:
            mydist=""
            mygain=""
            for x in range(len(np.array(dists)[idx2])):
                mydist=mydist+" "+str(round(np.array(dists)[idx2][x],2))
                mygain=mygain+" "+str(round(np.array(gain_factors)[idx2][x],2))
            plt.xlabel(r"Epoch %d (MJD %d, $\rm D_s$%s, $\rm G_f$%s)"%(i+1,days[i],mydist,mygain))
        plt.ylabel("Fluence(Jy ms)")
        plt.ylim([0,max(allfluence)*1.05])
    i+=1
    plt.subplot(len(days)//2+1,2,i+1)
    number, binwidth, patches=plt.hist(allfluence,
                                    bins ="auto"
                                    , histtype='step',color="deeppink",alpha=0.65)
    xrange=5*(max(number)//25+1)
    plt.yticks(np.arange(0,max(number)+xrange,xrange,dtype=int))
    plt.ylabel("Counts")
    plt.xlabel("Fluence(Jy ms)")


    if part:
        plt.suptitle(r'PSR %s (part); DM = %.1f $\rm pc\cdot \rm cm^{-3}$; P0 = %.6f s; PEpoch = %s'%(dirname,dm,p0,str(pepoch))+"\n Pulse Train(P0)",fontsize=20,fontname="DejaVu Serif")
    else:
        plt.suptitle(r'PSR %s; DM = %.1f $\rm pc\cdot \rm cm^{-3}$; P0 = %.6f s; PEpoch = %s'%(dirname,dm,p0,str(pepoch))+"\n Pulse Train(P0)",fontsize=20,fontname="DejaVu Serif")
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # rect=[left, bottom, right, top]
    fig.savefig(dirname+"_obs"+str(needmjd)+"_fluence.png",dpi=300,bbox_inches = 'tight')
    plt.close()


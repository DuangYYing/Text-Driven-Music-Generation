
import os
import numpy as np
path="E:\\jupyter\\data\\raw\\lmd-full_and_reddit_MIDI_dataset\\dev_5\\test4"#一个文件夹下多个npy文件
fp = open('word1.txt','w')
fmidi = open('word2.txt','w')
fduration = open('word3.txt','w')
#txtpath='./input_output'
#input_data = np.load(r"E:\jupyter\data\raw\lmd-full_and_reddit_MIDI_dataset\sentence_level_31\test.npy")
namelist=[x for x in os.listdir(path)]
for i in range( len(namelist) ):
	datapath=os.path.join(path,namelist[i])
	input_data = np.load(datapath) # (39, 2)
	data = input_data.reshape(1, -1)
	for key in data[0][2]:
		L1 =[s.lower() for s in key]
		fp.write(",".join(L1)+"\n")

	for note in data[0][1]:
		pitch2 = []
		for pitch in note:
			pitch2.append(pitch[0])

		fmidi.write(",".join(str(i) for i in pitch2)+"\n")

	for duration in data[0][1]:
		duration2 = []
		#print(duration)
		for dura in duration:
			#print (dura)
			duration2.append(dura[1])

		fduration.write(",".join(str(i) for i in duration2)+"\n")



        #for key in data[0][2]:
        #    fp.write(",".join(key)+"\n")
	#data = np.load(datapath).reshape([-1, 2])  # (39, 2)
	#input_data = np.load(datapath) # (39, 2)
	#data = input_data.reshape(1,-1)
    #print(data)
    #for key in data[0][2]:
    #    fp.write(",".join(key)+"\n")


	#np.savetxt('%s/%s.txt'%(txtpath,namelist[i]),data)
